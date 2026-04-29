"""Train TranscriptPromptedWhisper on KenSpeech x NLLB Sw->En pairs.

Usage on Kaggle 2x T4:
    python whisper_st/train.py \
        --base_model openai/whisper-small \
        --translations_dir /kaggle/working/hibiki-sw/translations/sw2en \
        --kenspeech_dir /kaggle/input/kenspeech-sw \
        --output_dir /kaggle/working/whisper_st_sw2en \
        --batch_size 8 --grad_accum 2 \
        --lr 1e-5 --epochs 3

Whisper-small is ~244M parameters; fits in fp16 on a single T4 with batch 8 and
3000-frame input features. Training time on KenSpeech (~5800 samples, 3 epochs):
roughly 4-6 hrs on T4. Use Whisper-base if speed matters more than quality.

The model produces both:
    - CTC logits over Whisper vocab (encoder side, supervises Sw transcript)
    - Decoder logits over Whisper vocab (supervises En translation, transcript-prompted)

Eval prints CTC-greedy transcript samples and decoder-greedy translation samples
periodically so you can sanity-check during training.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import random_split
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
)

# Repo-relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from whisper_st.dataset import KenSpeechSTDataset, TranscriptPromptedCollator
from whisper_st.model import TranscriptPromptedWhisper


class TranscriptPromptedTrainer(Seq2SeqTrainer):
    """Defensive compute_loss: bypass Trainer's label_smoother (which assumes
    the standard `labels` schema and would mishandle our masked transcript
    portion) and pass all batch keys directly to model.forward(). Required for
    correctness under both single-GPU and DDP/Accelerate launches.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="openai/whisper-small")
    parser.add_argument("--translations_dir", type=str, required=True,
                        help="Directory of NLLB translation JSONs (sw->en)")
    parser.add_argument("--kenspeech_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--ctc_loss_weight", type=float, default=0.3)
    parser.add_argument("--ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--eval_split", type=float, default=0.05,
                        help="Fraction of training data to hold out for eval")
    parser.add_argument("--max_audio_seconds", type=float, default=28.0)
    parser.add_argument("--max_label_tokens", type=int, default=384)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Mixed-precision mode. bf16 (default) requires Ampere/Ada/L4/A100/H100. "
                             "Use fp16 for older GPUs (T4); fp32 for debugging.")
    parser.add_argument("--lexicon_path", type=str, default=None,
                        help="Optional JSONL lexicon. If set, training-time hint injection "
                             "is enabled so the model learns to use lexicon hints at inference.")
    parser.add_argument("--hint_prob", type=float, default=0.5,
                        help="Probability per sample of injecting lexicon hints during training "
                             "(only takes effect when --lexicon_path is set).")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Processor + model
    print(f"Loading processor and model: {args.base_model}")
    processor = WhisperProcessor.from_pretrained(args.base_model, language="sw", task="transcribe")
    model = TranscriptPromptedWhisper.from_pretrained(
        args.base_model,
        ctc_loss_weight=args.ctc_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        ctc_blank_id=processor.tokenizer.pad_token_id or 0,
    )
    # Disable forced_decoder_ids and suppress_tokens because we pass full
    # decoder_input_ids ourselves. These are GENERATION params; in transformers
    # 4.46+ the save_pretrained validator rejects them being set on model.config
    # (must live on model.generation_config). We clear both locations.
    for key in ("forced_decoder_ids", "suppress_tokens", "begin_suppress_tokens"):
        if hasattr(model.config, key):
            setattr(model.config, key, None)
        if hasattr(model.generation_config, key):
            setattr(model.generation_config, key, None)

    # Dataset
    print("Building dataset...")
    full_ds = KenSpeechSTDataset(
        translations_dir=args.translations_dir,
        kenspeech_dir=args.kenspeech_dir,
        processor=processor,
        max_audio_seconds=args.max_audio_seconds,
        max_label_tokens=args.max_label_tokens,
        lexicon_path=args.lexicon_path,
        hint_prob=args.hint_prob,
    )

    n_eval = max(1, int(len(full_ds) * args.eval_split))
    n_train = len(full_ds) - n_eval
    train_ds, eval_ds = random_split(
        full_ds, [n_train, n_eval], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Splits: train={len(train_ds)}, eval={len(eval_ds)}")

    collator = TranscriptPromptedCollator(
        pad_token_id=processor.tokenizer.pad_token_id or 50257,
        ctc_blank_id=processor.tokenizer.pad_token_id or 0,
    )

    # Precision flags
    use_bf16 = args.precision == "bf16"
    use_fp16 = args.precision == "fp16"

    # Trainer args. transformers 4.46+ renamed `evaluation_strategy` -> `eval_strategy`;
    # we try the new name first and fall back to the old one for older versions.
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
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=args.save_total_limit,
        logging_steps=25,
        report_to=["none"],
        remove_unused_columns=False,    # we pass custom kwargs
        label_names=["labels", "transcript_labels", "transcript_label_lengths"],
        predict_with_generate=False,    # custom forward; eval CE only
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        ddp_find_unused_parameters=False,   # we use all params; faster DDP
    )
    try:
        train_args = Seq2SeqTrainingArguments(eval_strategy="steps", **common_kwargs)
    except TypeError:
        # Older transformers (<4.46) only accepts the old name
        train_args = Seq2SeqTrainingArguments(evaluation_strategy="steps", **common_kwargs)

    # transformers 4.46+ renamed Trainer's `tokenizer` arg to `processing_class`
    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    try:
        trainer = TranscriptPromptedTrainer(processing_class=processor.tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = TranscriptPromptedTrainer(tokenizer=processor.tokenizer, **trainer_kwargs)

    print("Starting training...")
    trainer.train()

    # Save final model + processor
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nDone. Final model saved to {final_dir}")

    # Save training config for reproducibility
    with open(final_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
