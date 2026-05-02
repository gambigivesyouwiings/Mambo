"""Pseudo-label unlabeled Sw audio with Whisper-large-v3 (the teacher).

Inputs: HuggingFace datasets containing Sw audio (Common Voice, FLEURS train).
Outputs:
    - <out_dir>/audio/<source>/<id>.wav  : 16 kHz mono WAVs (so we don't depend on HF cache)
    - <out_dir>/pseudo_labels.jsonl       : one JSON per audio with:
        {
          "source": "cv_sw" | "fleurs_sw_train",
          "id": "...",
          "audio_path": "<abs path>",
          "duration_s": float,
          "pseudo_label": "...",
          "avg_log_prob": float,        # length-normalized score from teacher's generate()
          "gold_label": "..."           # original transcript if dataset has one (for upper-bound experiment)
        }

Why we save audio to disk:
    Streaming the HF dataset at training time would re-download/re-decode for every epoch.
    Saving once to local WAV makes the student training pass IO-trivial.

Resume:
    If pseudo_labels.jsonl already exists we read off the set of (source, id) keys that
    are done and skip them. Safe to re-run after a crash.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# ---- Source dataset specs -----------------------------------------------------

# Each entry returns an iterable of dicts with: id, audio_array, sampling_rate, gold_label.
# Lazy-loaded so we don't import datasets unless needed.

def _iter_common_voice(version: str = "17_0", max_samples: Optional[int] = None) -> Iterable[Dict]:
    """Mozilla Common Voice Sw, train+dev splits (we hold out CV-Sw test for our own eval if needed)."""
    from datasets import load_dataset
    for split in ("train", "validation"):
        ds = load_dataset(
            f"mozilla-foundation/common_voice_{version}",
            "sw",
            split=split,
            trust_remote_code=True,
        )
        for i, s in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            yield {
                "id": f"{split}_{s.get('client_id', '')[:8]}_{i:06d}",
                "audio_array": np.asarray(s["audio"]["array"], dtype=np.float32),
                "sampling_rate": s["audio"]["sampling_rate"],
                "gold_label": (s.get("sentence") or "").strip(),
            }


def _iter_common_voice_local(local_dir: str, max_samples: Optional[int] = None) -> Iterable[Dict]:
    """Mozilla CV-Sw from a locally extracted dataset (Kaggle layout).

    Expected directory layout:
        local_dir/
            clips/*.mp3
            validated.tsv     # standard CV columns: client_id, path, sentence, ...
    """
    import csv
    import librosa

    root = Path(local_dir)
    tsv_path = root / "validated.tsv"
    clips_dir = root / "clips"

    if not tsv_path.exists():
        raise FileNotFoundError(f"Expected validated.tsv at {tsv_path}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"Expected clips/ at {clips_dir}")

    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        emitted = 0
        for row in reader:
            if max_samples is not None and emitted >= max_samples:
                break
            rel_path = (row.get("path") or "").strip()
            sentence = (row.get("sentence") or "").strip()
            if not rel_path or not sentence:
                continue
            audio_path = clips_dir / rel_path
            if not audio_path.exists():
                continue
            try:
                audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
            except Exception as e:
                print(f"  [skip] failed to load {audio_path}: {e}")
                continue
            yield {
                "id": Path(rel_path).stem,  # e.g. "common_voice_sw_12345"
                "audio_array": audio.astype(np.float32),
                "sampling_rate": 16000,
                "gold_label": sentence,
            }
            emitted += 1


def _iter_fleurs_sw_train(max_samples: Optional[int] = None) -> Iterable[Dict]:
    from datasets import load_dataset
    ds = load_dataset("google/fleurs", "sw_ke", split="train", trust_remote_code=True)
    for i, s in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        yield {
            "id": f"fleurs_train_{i:06d}",
            "audio_array": np.asarray(s["audio"]["array"], dtype=np.float32),
            "sampling_rate": s["audio"]["sampling_rate"],
            "gold_label": (s.get("transcription") or "").strip(),
        }


SOURCE_LOADERS = {
    "cv_sw": _iter_common_voice,
    "cv_sw_local": _iter_common_voice_local,
    "fleurs_sw_train": _iter_fleurs_sw_train,
}


# ---- Teacher inference --------------------------------------------------------

def _resample_if_needed(audio: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    import torchaudio
    t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr_in, sr_out)
    return t.squeeze(0).numpy()


@torch.no_grad()
def _teacher_transcribe(
    audio_16k: np.ndarray,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    max_new_tokens: int = 225,
) -> Tuple[str, float]:
    """Returns (text, length-normalized avg log-prob) for one audio."""
    feats = processor.feature_extractor(
        audio_16k, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device=device, dtype=next(model.parameters()).dtype)

    out = model.generate(
        input_features=feats,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        language="sw",
        task="transcribe",
    )

    seq = out.sequences[0]
    scores = getattr(out, "scores", None) or ()
    if scores:
        # generate() includes the decoder prompt at the front; output_scores covers only newly-generated tokens
        new_tokens = seq[-len(scores):]
        log_prob_sum = 0.0
        n_counted = 0
        eos_id = processor.tokenizer.eos_token_id
        for tok_id, score in zip(new_tokens.tolist(), scores):
            log_probs = torch.log_softmax(score[0], dim=-1)
            log_prob_sum += log_probs[tok_id].item()
            n_counted += 1
            if tok_id == eos_id:
                break
        avg_log_prob = log_prob_sum / max(1, n_counted)
    else:
        # output_scores was silently ignored by the transformers version we're on.
        # Fall back to whole-sequence decode without a confidence signal; the
        # confidence filter will be a no-op for these entries.
        new_tokens = seq
        avg_log_prob = 0.0

    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, avg_log_prob


# ---- Driver -------------------------------------------------------------------

def _read_done_keys(jsonl_path: Path) -> set:
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                done.add((e["source"], e["id"]))
            except Exception:
                continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for audio/ and pseudo_labels.jsonl")
    parser.add_argument("--teacher_model", default="openai/whisper-large-v3")
    parser.add_argument("--sources", nargs="+", default=["fleurs_sw_train", "cv_sw"],
                        choices=list(SOURCE_LOADERS.keys()))
    parser.add_argument("--cv_version", default="17_0",
                        help="Common Voice version suffix, e.g. '17_0' for cv_17_0 (HF-based loader only)")
    parser.add_argument("--cv_local_dir", default=None,
                        help="Local directory containing CV-Sw with clips/ + validated.tsv "
                             "(used by the cv_sw_local source).")
    parser.add_argument("--max_per_source", type=int, default=None,
                        help="Cap samples per source. None = all.")
    parser.add_argument("--max_audio_seconds", type=float, default=30.0,
                        help="Drop audios longer than this (Whisper's input cap).")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    audio_root = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(exist_ok=True)
    jsonl_path = out_dir / "pseudo_labels.jsonl"

    print("[pseudo_label.py rev=fix-bf16-and-scores]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    print(f"Loading teacher {args.teacher_model} on {device} ({args.precision})...")
    processor = WhisperProcessor.from_pretrained(args.teacher_model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.teacher_model, dtype=dtype
    ).to(device)
    model.eval()

    done = _read_done_keys(jsonl_path)
    print(f"Resume: {len(done)} samples already pseudo-labeled — will skip those.")

    # Open in append mode so we add to existing manifest on resume
    with open(jsonl_path, "a", encoding="utf-8") as out_f:
        for source in args.sources:
            print(f"\n=== Source: {source} ===")
            loader = SOURCE_LOADERS[source]
            kwargs = {"max_samples": args.max_per_source}
            if source == "cv_sw":
                kwargs["version"] = args.cv_version
            elif source == "cv_sw_local":
                if not args.cv_local_dir:
                    raise ValueError("--cv_local_dir is required when source is cv_sw_local")
                kwargs["local_dir"] = args.cv_local_dir

            n_done_this_source = 0
            n_skipped_dup = 0
            n_skipped_long = 0
            n_failed = 0
            for sample in loader(**kwargs):
                key = (source, sample["id"])
                if key in done:
                    n_skipped_dup += 1
                    continue

                # Resample to 16kHz once, before WAV write and before teacher
                audio = _resample_if_needed(sample["audio_array"], sample["sampling_rate"])
                duration_s = len(audio) / 16000.0
                if duration_s > args.max_audio_seconds:
                    n_skipped_long += 1
                    continue

                # Save audio
                src_dir = audio_root / source
                src_dir.mkdir(exist_ok=True)
                wav_path = src_dir / f"{sample['id']}.wav"
                if not wav_path.exists():
                    sf.write(str(wav_path), audio, 16000, subtype="PCM_16")

                try:
                    text, avg_lp = _teacher_transcribe(audio, model, processor, device)
                except Exception as e:
                    n_failed += 1
                    print(f"  [skip] {source}/{sample['id']}: {e}")
                    continue

                entry = {
                    "source": source,
                    "id": sample["id"],
                    "audio_path": str(wav_path),
                    "duration_s": round(duration_s, 3),
                    "pseudo_label": text,
                    "avg_log_prob": round(avg_lp, 4),
                    "gold_label": sample.get("gold_label", ""),
                }
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done_this_source += 1

                if n_done_this_source % args.log_every == 0:
                    print(f"  [{source}] {n_done_this_source} done | "
                          f"avg_lp={avg_lp:.3f} | "
                          f"text='{text[:60]}'")

            print(f"  Source done: +{n_done_this_source} new "
                  f"(skipped {n_skipped_dup} dup, {n_skipped_long} too long, "
                  f"{n_failed} errors)")

    print(f"\nAll done. Manifest: {jsonl_path}")


if __name__ == "__main__":
    main()
