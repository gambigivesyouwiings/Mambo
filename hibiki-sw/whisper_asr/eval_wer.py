"""WER + CER eval for Whisper ASR models on FLEURS sw_ke (or any audio_dir + refs.jsonl).

Usage:
    python whisper_asr/eval_wer.py \
        --models \
            vanilla=openai/whisper-small \
            ft_kenspeech=/data/asr_runs/ft_kenspeech_only/final \
            ft_pseudo_filtered=/data/asr_runs/ft_kenspeech_pseudo_filtered/final \
        --audio_dir /data/runs/fleurs_sw_ke_test/audio \
        --references_path /data/runs/fleurs_sw_ke_test/refs.jsonl \
        --output_path /data/asr_runs/results.txt \
        --show_examples 10

References JSONL format (already produced by whisper_st/run_overnight.sh Step 2):
    {"audio": "fleurs_sw_ke_00000.wav", "reference_sw": "...", ...}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# Lightweight Whisper-style normalization for fair WER comparison.
# We don't pull in `whisper-normalizer` to avoid extra deps; this covers the basics:
# lowercase, strip punctuation, collapse whitespace.
_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


@torch.no_grad()
def transcribe_one(
    audio: np.ndarray,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    max_new_tokens: int = 225,
) -> str:
    feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    out = model.generate(
        input_features=feats,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        language="sw",
        task="transcribe",
    )
    return processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if sr != 16000:
        import torchaudio
        t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, 16000)
        audio = t.squeeze(0).numpy()
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


def evaluate_model(
    model_path: str,
    refs: Dict[str, str],
    audio_dir: Path,
    device: str,
    max_samples: int = None,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Returns (predictions_by_audio, ordered_preds, ordered_refs)."""
    print(f"\nLoading {model_path}...")
    processor = WhisperProcessor.from_pretrained(model_path, language="sw", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    # Stable ordering keyed by audio filename
    audio_names = sorted(refs.keys())
    if max_samples:
        audio_names = audio_names[:max_samples]

    preds_by_audio = {}
    pred_list, ref_list = [], []
    for i, name in enumerate(audio_names):
        wav_path = audio_dir / name
        if not wav_path.exists():
            continue
        try:
            audio = load_audio(wav_path)
            pred = transcribe_one(audio, model, processor, device)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        preds_by_audio[name] = pred
        pred_list.append(pred)
        ref_list.append(refs[name])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(audio_names)}")

    # Free GPU mem before loading the next model
    del model
    torch.cuda.empty_cache()
    return preds_by_audio, pred_list, ref_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="name=path pairs; path can be HF hub id or local dir.")
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--references_path", required=True,
                        help="JSONL with {audio, reference_sw}")
    parser.add_argument("--output_path", required=True,
                        help="Results table (.txt). Predictions JSONL goes next to it.")
    parser.add_argument("--show_examples", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    import evaluate
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Load references
    refs: Dict[str, str] = {}
    with open(args.references_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("audio") and e.get("reference_sw"):
                    refs[e["audio"]] = e["reference_sw"]
            except Exception:
                continue
    print(f"Loaded {len(refs)} references")

    audio_dir = Path(args.audio_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds_dir = out_path.parent

    # Run each model
    name_to_preds: Dict[str, Dict[str, str]] = {}
    name_to_scores: Dict[str, Tuple[float, float]] = {}

    for spec in args.models:
        if "=" not in spec:
            raise ValueError(f"--models entries must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        preds_by_audio, pred_list, ref_list = evaluate_model(
            path, refs, audio_dir, device, max_samples=args.max_samples
        )
        name_to_preds[name] = preds_by_audio

        # Normalize for fair scoring
        norm_preds = [normalize(p) for p in pred_list]
        norm_refs = [normalize(r) for r in ref_list]
        wer = 100.0 * wer_metric.compute(predictions=norm_preds, references=norm_refs)
        cer = 100.0 * cer_metric.compute(predictions=norm_preds, references=norm_refs)
        name_to_scores[name] = (wer, cer)
        print(f"  {name}:  WER={wer:.2f}  CER={cer:.2f}  (n={len(pred_list)})")

        # Save per-model predictions
        with open(preds_dir / f"preds_{name}.jsonl", "w", encoding="utf-8") as f:
            for audio, pred in preds_by_audio.items():
                f.write(json.dumps({"audio": audio, "prediction": pred,
                                    "reference": refs[audio]}, ensure_ascii=False) + "\n")

    # Write results table
    common_audios = sorted(set.intersection(*[set(p.keys()) for p in name_to_preds.values()]))
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write(f"References: {len(refs)}\n")
        for name, preds in name_to_preds.items():
            out_f.write(f"  {name}: {len(preds)} predictions\n")
        out_f.write(f"\nScoring on {len(common_audios)} audios common to all systems\n\n")
        out_f.write(f"{'System':<35} {'WER':>8} {'CER':>8}\n")
        out_f.write("-" * 55 + "\n")
        for name in name_to_preds:
            wer, cer = name_to_scores[name]
            out_f.write(f"{name:<35} {wer:>8.2f} {cer:>8.2f}\n")

        if args.show_examples > 0 and common_audios:
            out_f.write("\n=== First {} examples ===\n".format(min(args.show_examples, len(common_audios))))
            for audio in common_audios[: args.show_examples]:
                out_f.write(f"\n[{audio}]\n  REF: {refs[audio]}\n")
                for name in name_to_preds:
                    pred = name_to_preds[name].get(audio, "")
                    out_f.write(f"  {name:<30}: {pred}\n")

    print(f"\nResults -> {out_path}")
    # Echo the table to stdout too
    with open(out_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
