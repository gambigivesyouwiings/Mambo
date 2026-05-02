"""Fine-tune Whisper for Swahili ASR.

Standard recipe — uses HF's Seq2SeqTrainer with `predict_with_generate=True` so we
can compute WER on the eval split during training. No custom architecture, no CTC head.

Variants are produced by toggling --pseudo_labels_path:
  - none                                : KenSpeech-only baseline (supervised)
  - raw pseudo_labels.jsonl             : KenSpeech ∪ unfiltered pseudo (naive baseline)
  - filtered pseudo_labels.jsonl        : KenSpeech ∪ filtered pseudo (novel method)
  - pseudo with --use_gold (filter step): KenSpeech ∪ gold-from-CV-Sw (upper bound)

Usage on g6.12xlarge / 4x L4:
    accelerate launch --num_processes 4 --mixed_precision bf16 \
        whisper_asr/train.py \
        --base_model openai/whisper-small \
        --kenspeech_dir /home/ec2-user/data/kenspeech/kenspeech-sw \
        --output_dir /home/ec2-user/data/asr_runs/ft_kenspeech_only \
        --batch_size 16 --grad_accum 1 \
        --lr 1e-5 --epochs 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from whisper_asr.dataset import SwASRDataset, WhisperASRCollator


def build_compute_metrics(processor):
    """Returns a compute_metrics fn that decodes preds + labels and computes WER."""
    import evaluate
    wer_metric = evaluate.load("wer")
    pad_token_id = processor.tokenizer.pad_token_id

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # Replace -100 with pad so we can decode
        label_ids = np.where(label_ids != -100, label_ids, pad_token_id)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # WER metric crashes on empty references; filter to non-empty pairs and
        # substitute a single space for empty predictions (counts as full-deletion).
        pairs = [(p if p.strip() else " ", r) for p, r in zip(pred_str, label_str) if r.strip()]
        if not pairs:
            return {"wer": 100.0}
        preds, refs = zip(*pairs)
        wer = 100.0 * wer_metric.compute(predictions=list(preds), references=list(refs))
        return {"wer": wer}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="openai/whisper-small")
    parser.add_argument("--kenspeech_dir", required=True)
    parser.add_argument("--pseudo_labels_path", default=None,
                        help="Optional JSONL of pseudo (or gold) labels to add as training data.")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_split", type=float, default=0.05)
    parser.add_argument("--max_audio_seconds", type=float, default=28.0)
    parser.add_argument("--max_label_tokens", type=int, default=448)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen_max_new_tokens", type=int, default=225)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("[train.py rev=strip-bos-and-empty-skip]")
    print(f"Loading processor and model: {args.base_model}")
    processor = WhisperProcessor.from_pretrained(args.base_model, language="sw", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    # We use HF's standard Whisper training — let it set forced_decoder_ids from the
    # processor language/task config rather than us interfering. Just clear any
    # legacy generation params on the config so save_pretrained doesn't choke.
    for key in ("forced_decoder_ids", "suppress_tokens", "begin_suppress_tokens"):
        if hasattr(model.config, key):
            setattr(model.config, key, None)
    # Make generation_config language/task-aware so eval generation produces Sw transcripts.
    model.generation_config.language = "swahili"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    print("Building dataset...")
    full_ds = SwASRDataset(
        kenspeech_dir=args.kenspeech_dir,
        processor=processor,
        pseudo_labels_path=args.pseudo_labels_path,
        max_audio_seconds=args.max_audio_seconds,
        max_label_tokens=args.max_label_tokens,
    )

    n_eval = max(1, int(len(full_ds) * args.eval_split))
    n_train = len(full_ds) - n_eval
    train_ds, eval_ds = random_split(
        full_ds, [n_train, n_eval], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Splits: train={len(train_ds)}, eval={len(eval_ds)}")

    collator = WhisperASRCollator(pad_token_id=processor.tokenizer.pad_token_id or 50257)

    use_bf16 = args.precision == "bf16"
    use_fp16 = args.precision == "fp16"

    common_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_steps=400,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=args.save_total_limit,
        logging_steps=25,
        report_to=["none"],
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=True,
        generation_max_length=args.gen_max_new_tokens,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    try:
        train_args = Seq2SeqTrainingArguments(eval_strategy="steps", **common_kwargs)
    except TypeError:
        train_args = Seq2SeqTrainingArguments(evaluation_strategy="steps", **common_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
    )
    try:
        trainer = Seq2SeqTrainer(processing_class=processor.tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Seq2SeqTrainer(tokenizer=processor.tokenizer, **trainer_kwargs)

    print("Starting training...")
    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nDone. Final model saved to {final_dir}")

    with open(final_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
