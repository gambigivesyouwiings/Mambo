"""SeamlessM4T-v2-large speech-to-text translation evaluation on FLEURS.

Runs Sw->En and En->Sw S2T translation (Seamless's strongest task on Sw -- the
S2ST speech-output decoder is significantly weaker than its S2T text decoder)
and reports corpus BLEU + chrF against the FLEURS parallel transcripts.

For Sw->En: reads our locally-saved FLEURS sw_ke audio (built earlier by the
S2ST pipeline) so we evaluate on the exact same 487-utterance test split the
cascade was scored on. For En->Sw: streams FLEURS en_us audio via HuggingFace.

Outputs:
  - <out_dir>/seamless_sw2en.jsonl   : per-utterance hyp + ref + audio key
  - <out_dir>/seamless_en2sw.jsonl
  - <out_dir>/seamless_results.md    : paste-ready markdown table

Usage:
    python whisper_asr/eval_seamless.py \
        --refs_path /home/ec2-user/data/runs/fleurs_sw_ke_test/refs.jsonl \
        --audio_dir /home/ec2-user/data/runs/fleurs_sw_ke_test/audio \
        --out_dir /home/ec2-user/data/asr_runs/seamless_eval

Notes:
  - SeamlessM4T-v2-large is ~2.3B parameters. First run downloads ~10 GB.
  - bf16 inference fits comfortably in 24 GB VRAM (A10G).
  - Single-sample inference at ~2-4 sec per utterance; full eval ~30-60 min.
  - Resume support: skips per-direction outputs that already exist.

Dependencies: pip install sacrebleu sentencepiece protobuf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model


# Seamless uses 3-letter ISO 639-3 codes
SW_CODE = "swh"
EN_CODE = "eng"


# ---- Inference helpers ------------------------------------------------------

@torch.no_grad()
def s2t_translate(
    audio_array: np.ndarray,
    sampling_rate: int,
    tgt_lang: str,
    processor,
    model,
    device: str,
) -> str:
    """Run one S2T translation. Returns decoded target text."""
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32, copy=False)

    inputs = processor(
        audios=audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).to(device)

    out = model.generate(
        **inputs,
        tgt_lang=tgt_lang,
        generate_speech=False,
        num_beams=4,
    )
    # generate(generate_speech=False) returns a tuple whose first element is text token IDs
    if isinstance(out, tuple):
        token_ids = out[0]
    else:
        token_ids = out
    text = processor.decode(token_ids[0].tolist(), skip_special_tokens=True)
    return text.strip()


def _resample_if_needed(audio: np.ndarray, sr_in: int, sr_out: int = 16000) -> Tuple[np.ndarray, int]:
    if sr_in == sr_out:
        return audio, sr_in
    import torchaudio
    t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr_in, sr_out)
    return t.squeeze(0).numpy(), sr_out


# ---- Direction evaluators ---------------------------------------------------

def evaluate_sw_to_en(
    refs_path: Path,
    audio_dir: Path,
    processor,
    model,
    device: str,
    out_path: Path,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Sw audio (local FLEURS sw_ke wavs) -> En text via Seamless S2T."""
    # Load (audio_path, ref_en) pairs from refs.jsonl
    items: List[Tuple[Path, str]] = []
    with open(refs_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            audio = e.get("audio")
            ref_en = (e.get("reference_en") or "").strip()
            if not audio or not ref_en:
                continue
            wav_path = audio_dir / audio
            if not wav_path.exists():
                continue
            items.append((wav_path, ref_en))

    if max_samples is not None:
        items = items[:max_samples]
    print(f"  Sw->En: {len(items)} utterances to translate")

    hyps, refs = [], []
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, (wav_path, ref) in enumerate(items):
            audio, sr = sf.read(str(wav_path))
            audio, sr = _resample_if_needed(audio, sr)
            try:
                hyp = s2t_translate(audio, sr, EN_CODE, processor, model, device)
            except Exception as e:
                print(f"  [skip] {wav_path.name}: {e}")
                continue
            hyps.append(hyp)
            refs.append(ref)
            out_f.write(json.dumps({
                "audio": wav_path.name,
                "hypothesis": hyp,
                "reference": ref,
            }, ensure_ascii=False) + "\n")
            out_f.flush()
            if (i + 1) % 25 == 0:
                print(f"  Sw->En  {i+1}/{len(items)}  | hyp: {hyp[:70]}")
    return hyps, refs


def evaluate_en_to_sw(
    processor,
    model,
    device: str,
    out_path: Path,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """En audio (FLEURS en_us test, streamed from HF) -> Sw text via Seamless S2T.

    Restricts to en_us samples whose `id` also appears in FLEURS sw_ke test, so
    we score against the parallel Sw transcript.
    """
    from datasets import load_dataset
    print("  Loading FLEURS en_us + sw_ke test splits via HF (one-time download)...")
    en_ds = load_dataset("google/fleurs", "en_us", split="test", trust_remote_code=True)
    sw_ds = load_dataset("google/fleurs", "sw_ke", split="test", trust_remote_code=True)
    sw_by_id = {s["id"]: s["transcription"] for s in sw_ds}

    samples = [s for s in en_ds if s["id"] in sw_by_id]
    if max_samples is not None:
        samples = samples[:max_samples]
    print(f"  En->Sw: {len(samples)} parallel utterances to translate")

    hyps, refs = [], []
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, s in enumerate(samples):
            audio = np.asarray(s["audio"]["array"], dtype=np.float32)
            sr = s["audio"]["sampling_rate"]
            audio, sr = _resample_if_needed(audio, sr)
            ref = sw_by_id[s["id"]]
            try:
                hyp = s2t_translate(audio, sr, SW_CODE, processor, model, device)
            except Exception as e:
                print(f"  [skip] en_us id={s['id']}: {e}")
                continue
            hyps.append(hyp)
            refs.append(ref)
            out_f.write(json.dumps({
                "id": s["id"],
                "hypothesis": hyp,
                "reference": ref,
            }, ensure_ascii=False) + "\n")
            out_f.flush()
            if (i + 1) % 25 == 0:
                print(f"  En->Sw  {i+1}/{len(samples)}  | hyp: {hyp[:70]}")
    return hyps, refs


# ---- Driver -----------------------------------------------------------------

def _resume_load(out_path: Path) -> Tuple[List[str], List[str]]:
    hyps, refs = [], []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                hyps.append(e["hypothesis"])
                refs.append(e["reference"])
            except Exception:
                continue
    return hyps, refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refs_path",
        default="/home/ec2-user/data/runs/fleurs_sw_ke_test/refs.jsonl",
    )
    parser.add_argument(
        "--audio_dir",
        default="/home/ec2-user/data/runs/fleurs_sw_ke_test/audio",
    )
    parser.add_argument("--out_dir", default="/home/ec2-user/data/asr_runs/seamless_eval")
    parser.add_argument("--model", default="facebook/seamless-m4t-v2-large")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["sw2en", "en2sw"],
        choices=["sw2en", "en2sw"],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.precision
    ]
    print(f"Loading {args.model} on {device} ({args.precision}) -- this may take a few minutes on first run...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = SeamlessM4Tv2Model.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    try:
        import sacrebleu
    except ImportError:
        raise SystemExit("sacrebleu not installed. Run: pip install sacrebleu")

    results: Dict[str, Tuple[float, float, int]] = {}

    if "sw2en" in args.directions:
        print("\n=== Sw -> En S2T ===")
        out_path = out_dir / "seamless_sw2en.jsonl"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  Resume: {out_path} exists, loading...")
            hyps, refs = _resume_load(out_path)
        else:
            hyps, refs = evaluate_sw_to_en(
                Path(args.refs_path),
                Path(args.audio_dir),
                processor,
                model,
                device,
                out_path,
                max_samples=args.max_samples,
            )
        if hyps:
            bleu = sacrebleu.corpus_bleu(hyps, [refs])
            chrf = sacrebleu.corpus_chrf(hyps, [refs])
            results["sw2en"] = (bleu.score, chrf.score, len(hyps))
            print(f"  Sw->En  BLEU={bleu.score:.2f}  chrF={chrf.score:.2f}  (n={len(hyps)})")

    if "en2sw" in args.directions:
        print("\n=== En -> Sw S2T ===")
        out_path = out_dir / "seamless_en2sw.jsonl"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  Resume: {out_path} exists, loading...")
            hyps, refs = _resume_load(out_path)
        else:
            hyps, refs = evaluate_en_to_sw(
                processor,
                model,
                device,
                out_path,
                max_samples=args.max_samples,
            )
        if hyps:
            bleu = sacrebleu.corpus_bleu(hyps, [refs])
            chrf = sacrebleu.corpus_chrf(hyps, [refs])
            results["en2sw"] = (bleu.score, chrf.score, len(hyps))
            print(f"  En->Sw  BLEU={bleu.score:.2f}  chrF={chrf.score:.2f}  (n={len(hyps)})")

    # Markdown
    rows = ["| Direction | BLEU | chrF | n |", "|---|---|---|---|"]
    label = {"sw2en": "Sw -> En", "en2sw": "En -> Sw"}
    for direction in ["sw2en", "en2sw"]:
        if direction not in results:
            continue
        bleu, chrf, n = results[direction]
        rows.append(f"| {label[direction]} | {bleu:.2f} | {chrf:.2f} | {n} |")

    md = (
        "# SeamlessM4T-v2-large S2T Results on FLEURS sw_ke <-> en_us\n\n"
        "Speech-to-text translation only (S2ST speech-output is significantly weaker on Sw\n"
        "and is omitted). Beam search with beam=4. Scored with sacrebleu corpus BLEU + chrF.\n\n"
        "## For Table 10 (Section 5.6)\n\n"
        + "\n".join(rows) + "\n"
    )
    md_path = out_dir / "seamless_results.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\nWrote {md_path}")
    print("\n--- Preview ---\n")
    print(md)


if __name__ == "__main__":
    main()
