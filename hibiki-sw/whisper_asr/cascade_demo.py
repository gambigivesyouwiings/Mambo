"""Cascade S2S demo: end-to-end Whisper -> NLLB -> MMS-TTS, producing audio pairs.

Mirrors notebooks/07_cascade_s2s_demo.ipynb but as a headless CLI script so it
runs cleanly on the A10G over SSH (no Jupyter port-forwarding needed).

Outputs (under --out_dir):
    source/sample_NN.wav                    -- original source audio
    target/sample_NN_<asr_variant>.wav      -- synthesised target audio per variant
    transcripts.jsonl                       -- machine-readable per-sample texts
    index.md                                -- human-readable side-by-side index

Usage on the A10G (Sw->En, all four fine-tuned variants + vanilla):

    python whisper_asr/cascade_demo.py \
        --direction sw2en \
        --num_samples 10 \
        --asr_root /home/ec2-user/data/asr_runs \
        --out_dir /home/ec2-user/data/asr_runs/cascade_demo

To skip variants whose path is missing, the script silently drops them.
The vanilla openai/whisper-small baseline is always included for delta comparison.

Dependencies: torch, transformers, faster-whisper, soundfile, datasets, torchaudio.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taf


VARIANT_NAMES = [
    "ft_kenspeech_only",
    "ft_kenspeech_pseudo_raw",
    "ft_kenspeech_pseudo_filtered",
    "ft_kenspeech_gold_upper_bound",
]


# ---- GPU bookkeeping --------------------------------------------------------

def free_gpu(label: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU{(' ' + label) if label else ''}: {used:.1f}/{total:.1f} GB")


# ---- Audio collection -------------------------------------------------------

def collect_fleurs_samples(
    direction: str, num_samples: int, out_dir: Path
) -> List[dict]:
    """Pick parallel FLEURS samples and write source WAVs to disk.

    Returns a list of dicts with source path + gold text + parallel reference.
    """
    from datasets import load_dataset

    if direction == "sw2en":
        src_subset, tgt_subset = "sw_ke", "en_us"
    else:
        src_subset, tgt_subset = "en_us", "sw_ke"

    print(f"Loading FLEURS {src_subset} test split...")
    src_ds = load_dataset("google/fleurs", src_subset, split="test", trust_remote_code=True)
    print(f"Loading FLEURS {tgt_subset} test split (for parallel target reference)...")
    tgt_ds = load_dataset("google/fleurs", tgt_subset, split="test", trust_remote_code=True)
    tgt_by_id = {s["id"]: s["transcription"] for s in tgt_ds}

    src_dir = out_dir / "source"
    src_dir.mkdir(parents=True, exist_ok=True)

    picks = []
    for i, s in enumerate(src_ds):
        if s["id"] in tgt_by_id:
            picks.append((i, s))
        if len(picks) >= num_samples:
            break

    samples = []
    for n, (i, s) in enumerate(picks):
        wav = np.asarray(s["audio"]["array"], dtype=np.float32)
        sr = s["audio"]["sampling_rate"]
        wav_path = src_dir / f"sample_{n:02d}.wav"
        sf.write(str(wav_path), wav, sr)
        samples.append({
            "idx": n,
            "fleurs_id": s["id"],
            "source_wav": str(wav_path),
            "source_text_gold": s["transcription"],
            "reference_target": tgt_by_id[s["id"]],
        })
    print(f"Collected {len(samples)} FLEURS parallel samples")
    return samples


def collect_dir_samples(wav_dir: Path, num_samples: int, out_dir: Path) -> List[dict]:
    """Pick the first N WAVs from a directory and copy them as source samples."""
    src_dir = out_dir / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(wav_dir.glob("*.wav"))[:num_samples]
    samples = []
    for n, p in enumerate(wavs):
        wav, sr = sf.read(str(p))
        out_path = src_dir / f"sample_{n:02d}.wav"
        sf.write(str(out_path), wav, sr)
        samples.append({
            "idx": n,
            "fleurs_id": None,
            "source_wav": str(out_path),
            "source_text_gold": None,
            "reference_target": None,
            "source_basename": p.name,
        })
    print(f"Collected {len(samples)} samples from {wav_dir}")
    return samples


# ---- ASR --------------------------------------------------------------------

def load_asr(ckpt: str, device: str):
    """Returns (backend, model, processor_or_none).

    Local fine-tuned dirs go straight to transformers. Vanilla HF ids try
    faster-whisper first (CTranslate2, faster) with a transformers fallback.
    """
    is_local = os.path.isdir(ckpt)
    if not is_local:
        try:
            from faster_whisper import WhisperModel
            wm = WhisperModel(
                ckpt,
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="int8" if not torch.cuda.is_available() else "float16",
            )
            return ("faster_whisper", wm, None)
        except Exception as e:
            print(f"  faster-whisper unavailable for {ckpt} ({e}); using transformers")

    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(ckpt)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(ckpt, torch_dtype=dtype).to(device)
    model.eval()
    return ("transformers", model, proc)


def transcribe(
    backend: str, model, proc, wav: np.ndarray, sr: int, source_lang: str, device: str
) -> str:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = taf.resample(torch.from_numpy(wav.astype(np.float32)), sr, 16000).numpy()
    wav = wav.astype(np.float32)

    if backend == "faster_whisper":
        segments, _ = model.transcribe(wav, language=source_lang, task="transcribe", beam_size=1)
        return "".join(seg.text for seg in segments).strip()

    inp = proc(wav, sampling_rate=16000, return_tensors="pt")
    feats = inp.input_features.to(device, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        ids = model.generate(
            feats,
            language=source_lang, task="transcribe",
            num_beams=1, do_sample=False, max_new_tokens=225,
        )
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ---- Driver -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--direction", choices=["sw2en", "en2sw"], default="sw2en")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--asr_root", default="/home/ec2-user/data/asr_runs",
                   help="Root containing <variant>/final/ subdirs")
    p.add_argument("--out_dir", default="/home/ec2-user/data/asr_runs/cascade_demo",
                   help="Output directory; <direction> will be appended")
    p.add_argument("--source_dir", default=None,
                   help="Optional WAV directory to use instead of FLEURS test")
    p.add_argument("--mt_model", default="facebook/nllb-200-distilled-1.3B")
    p.add_argument("--mt_beam_size", type=int, default=4)
    p.add_argument("--variants", nargs="*", default=None,
                   help="Override variant list (default: all four fine-tuned + vanilla)")
    args = p.parse_args()

    out_dir = Path(args.out_dir) / args.direction
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "target").mkdir(parents=True, exist_ok=True)

    direction = args.direction
    if direction == "sw2en":
        source_lang, target_lang = "sw", "en"
        src_nllb, tgt_nllb = "swh_Latn", "eng_Latn"
        mms_lang = "eng"
    else:
        source_lang, target_lang = "en", "sw"
        src_nllb, tgt_nllb = "eng_Latn", "swh_Latn"
        mms_lang = "swh"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Direction:   {direction}  ({source_lang} -> {target_lang})")
    print(f"Device:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"ASR root:    {args.asr_root}")
    print(f"Output dir:  {out_dir}")

    # Build ASR variant list
    asr_root = Path(args.asr_root)
    variant_names = args.variants if args.variants else VARIANT_NAMES
    asr_variants: List[Tuple[str, str]] = [("vanilla_small", "openai/whisper-small")]
    for name in variant_names:
        path = asr_root / name / "final"
        if path.is_dir():
            asr_variants.append((name, str(path)))
        else:
            print(f"  [skip] {name}: {path} not found")
    print(f"ASR variants ({len(asr_variants)}): {[v[0] for v in asr_variants]}")
    if len(asr_variants) == 1:
        print("WARNING: only the vanilla variant is being used. "
              "Check --asr_root and that <variant>/final/ subdirs exist.")

    # Step 1: collect source samples
    if args.source_dir:
        samples = collect_dir_samples(Path(args.source_dir), args.num_samples, out_dir)
    else:
        samples = collect_fleurs_samples(direction, args.num_samples, out_dir)

    # Step 2: ASR for each variant
    asr_preds: Dict[str, Dict[int, str]] = {name: {} for name, _ in asr_variants}
    for name, ckpt in asr_variants:
        print(f"\n=== ASR: {name}  ({ckpt}) ===")
        backend, model, proc = load_asr(ckpt, device)
        print(f"  backend: {backend}")
        for s in samples:
            wav, sr = sf.read(s["source_wav"])
            text = transcribe(backend, model, proc, wav, sr, source_lang, device)
            asr_preds[name][s["idx"]] = text
            print(f"  [{s['idx']:02d}] {text[:120]}")
        del model
        if proc is not None:
            del proc
        free_gpu(f"after {name}")

    # Step 3: MT for each variant's transcripts
    print(f"\n=== MT: {args.mt_model} ===")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    mt_tok = AutoTokenizer.from_pretrained(args.mt_model, src_lang=src_nllb)
    mt_dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(args.mt_model, dtype=mt_dtype).to(device)
    mt_model.eval()
    tgt_id = mt_tok.convert_tokens_to_ids(tgt_nllb)
    free_gpu("after MT load")

    mt_preds: Dict[str, Dict[int, str]] = {name: {} for name, _ in asr_variants}
    for name, _ in asr_variants:
        print(f"  [{name}]")
        for s in samples:
            sw = asr_preds[name][s["idx"]] or " "
            enc = mt_tok(sw, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = mt_model.generate(
                    **enc,
                    forced_bos_token_id=tgt_id,
                    num_beams=args.mt_beam_size, max_new_tokens=256,
                )
            en = mt_tok.batch_decode(out, skip_special_tokens=True)[0].strip()
            mt_preds[name][s["idx"]] = en
            print(f"    [{s['idx']:02d}] {en[:120]}")
    del mt_model, mt_tok
    free_gpu("after MT")

    # Step 4: TTS for each variant
    print(f"\n=== TTS: facebook/mms-tts-{mms_lang} ===")
    from transformers import VitsModel, VitsTokenizer
    tts_name = f"facebook/mms-tts-{mms_lang}"
    tts_tok = VitsTokenizer.from_pretrained(tts_name)
    tts_model = VitsModel.from_pretrained(tts_name).to(device)
    tts_model.eval()
    tts_sr = tts_model.config.sampling_rate
    free_gpu("after TTS load")

    target_wavs: Dict[str, Dict[int, str]] = {name: {} for name, _ in asr_variants}
    target_dir = out_dir / "target"
    for name, _ in asr_variants:
        print(f"  [{name}]")
        for s in samples:
            en = mt_preds[name][s["idx"]] or " "
            inputs = tts_tok(en, return_tensors="pt").to(device)
            with torch.no_grad():
                out = tts_model(**inputs).waveform
            wav = out.squeeze().cpu().float().numpy()
            peak = float(np.max(np.abs(wav)))
            if peak > 0:
                wav = wav * (0.95 / peak)
            out_path = target_dir / f"sample_{s['idx']:02d}_{name}.wav"
            sf.write(str(out_path), wav.astype(np.float32), tts_sr)
            target_wavs[name][s["idx"]] = str(out_path)
            dur = len(wav) / tts_sr
            print(f"    [{s['idx']:02d}] {dur:.1f}s -> {out_path.name}")
    del tts_model, tts_tok
    free_gpu("after TTS")

    # Step 5: write transcripts.jsonl + index.md
    transcripts_path = out_dir / "transcripts.jsonl"
    with open(transcripts_path, "w", encoding="utf-8") as f:
        for s in samples:
            entry = {
                "idx": s["idx"],
                "fleurs_id": s.get("fleurs_id"),
                "source_wav": os.path.relpath(s["source_wav"], out_dir),
                "source_text_gold": s.get("source_text_gold"),
                "reference_target": s.get("reference_target"),
                "asr": {name: asr_preds[name][s["idx"]] for name, _ in asr_variants},
                "mt":  {name: mt_preds[name][s["idx"]]  for name, _ in asr_variants},
                "target_wavs": {name: os.path.relpath(target_wavs[name][s["idx"]], out_dir)
                                for name, _ in asr_variants},
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nWrote {transcripts_path}")

    md_lines = [f"# Cascade S2S Demo -- {direction}", ""]
    for s in samples:
        md_lines.append(f"## Sample {s['idx']:02d}  (fleurs_id={s.get('fleurs_id')})")
        if s.get("source_text_gold"):
            md_lines.append(f"- **Source gold ({source_lang}):** {s['source_text_gold']}")
        md_lines.append(f"- **Source audio:** [`source/{Path(s['source_wav']).name}`](source/{Path(s['source_wav']).name})")
        for name, _ in asr_variants:
            md_lines.append(f"- **[{name}] ASR:** {asr_preds[name][s['idx']]}")
            md_lines.append(f"- **[{name}] MT:** {mt_preds[name][s['idx']]}")
            tgt_rel = Path(target_wavs[name][s["idx"]]).name
            md_lines.append(f"- **[{name}] TTS:** [`target/{tgt_rel}`](target/{tgt_rel})")
        if s.get("reference_target"):
            md_lines.append(f"- **Target reference ({target_lang}):** {s['reference_target']}")
        md_lines.append("")
    index_path = out_dir / "index.md"
    index_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {index_path}")

    print(f"\nDone. {len(samples)} samples x {len(asr_variants)} ASR variants -> {out_dir}")
    print(f"Pull to your laptop:  scp -r ec2:{out_dir} ./cascade_demo_{direction}")


if __name__ == "__main__":
    main()
