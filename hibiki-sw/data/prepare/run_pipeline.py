"""End-to-end data preparation pipeline orchestrator.

Runs the full offline data processing chain:
    1. Whisper transcription (GPU)
    2. MADLAD translation (GPU)
    3. Contextual alignment (GPU — uses MADLAD encoder)
    4. Silence insertion (CPU)

Designed for Google Colab with a T4 GPU. Saves all outputs to Google Drive
for persistence across Colab sessions.

Usage:
    # Full pipeline for Swahili source audio (Sw→En direction)
    python data/prepare/run_pipeline.py \
        --source common_voice \
        --source_lang sw \
        --target_lang en \
        --base_dir /content/drive/MyDrive/hibiki-sw \
        --whisper_model medium \
        --max_samples 50000 \
        --step all

    # Run only specific steps (for resuming)
    python data/prepare/run_pipeline.py \
        --source common_voice \
        --source_lang sw \
        --target_lang en \
        --base_dir /content/drive/MyDrive/hibiki-sw \
        --step translate  # only run translation
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def step_transcribe(args):
    """Step 1: Whisper transcription with word timestamps."""
    from data.prepare.transcribe_whisper import WhisperTranscriber, process_common_voice

    print("=" * 60)
    print(f"STEP 1: Transcribe {args.source_lang} audio with Whisper")
    print("=" * 60)

    output_dir = os.path.join(
        args.base_dir, "transcriptions", args.source_lang
    )

    transcriber = WhisperTranscriber(
        model_size=args.whisper_model,
        device=args.device,
        compute_type="float16" if args.device == "cuda" else "int8",
    )

    n = process_common_voice(
        lang=args.source_lang,
        split="train",
        output_dir=output_dir,
        transcriber=transcriber,
        max_samples=args.max_samples,
        min_duration=1.0,
        max_duration=30.0,
        resume_from=args.resume_from,
    )

    print(f"Transcribed {n} samples -> {output_dir}\n")
    return output_dir


def step_translate(args):
    """Step 2: MADLAD-400 translation."""
    from data.prepare.translate_madlad import MADLADTranslator, translate_transcriptions

    print("=" * 60)
    print(f"STEP 2: Translate {args.source_lang} → {args.target_lang}")
    print("=" * 60)

    input_dir = os.path.join(
        args.base_dir, "transcriptions", args.source_lang
    )
    direction = f"{args.source_lang}2{args.target_lang}"
    output_dir = os.path.join(args.base_dir, "translations", direction)

    translator = MADLADTranslator(
        device=args.device,
        dtype="float16" if args.device == "cuda" else "float32",
    )

    n = translate_transcriptions(
        input_dir=input_dir,
        output_dir=output_dir,
        translator=translator,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        resume_from=args.resume_from,
    )

    print(f"Translated {n} samples -> {output_dir}\n")
    return output_dir


def step_align(args):
    """Step 3: Contextual alignment using MADLAD perplexity."""
    from data.contextual_align import batch_contextual_align

    print("=" * 60)
    print(f"STEP 3: Contextual alignment ({args.source_lang} → {args.target_lang})")
    print("=" * 60)

    direction = f"{args.source_lang}2{args.target_lang}"
    trans_dir = Path(args.base_dir) / "translations" / direction
    align_dir = Path(args.base_dir) / "alignments" / direction
    align_dir.mkdir(parents=True, exist_ok=True)

    # Load translation JSONs
    json_files = sorted(
        f for f in trans_dir.glob("*.json")
        if f.name != "index.jsonl"
    )

    if args.max_samples:
        json_files = json_files[:args.max_samples]

    print(f"Processing {len(json_files)} translation files...")

    # Batch processing for efficiency
    BATCH = 50
    processed = 0

    for batch_start in range(0, len(json_files), BATCH):
        batch_files = json_files[batch_start : batch_start + BATCH]
        pairs = []
        file_data = []

        for jp in batch_files:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            src_text = data["source_text"]
            tgt_text = data["translated_text"]
            if src_text.strip() and tgt_text.strip():
                pairs.append((src_text, tgt_text))
                file_data.append((jp, data))

        if not pairs:
            continue

        # Compute alignments
        try:
            alignments = batch_contextual_align(
                pairs,
                device=args.device,
                min_lag=2.0,
            )
        except Exception as e:
            print(f"  Alignment error at batch {batch_start}: {e}")
            continue

        # Save alignment results
        for (jp, data), alignment in zip(file_data, alignments):
            result = {
                "source_file": data.get("source_file", jp.name),
                "source_text": data["source_text"],
                "translated_text": data["translated_text"],
                "source_words": data.get("source_words", []),
                "alignment": alignment,  # list of (src_idx, tgt_idx)
                "duration": data.get("audio_duration", data.get("duration", 0)),
                "sample_idx": data.get("sample_idx", -1),
            }

            out_path = align_dir / jp.name.replace(f"_{args.target_lang}.json", "_aligned.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            processed += 1

        if processed % 500 == 0:
            print(f"  Aligned: {processed}/{len(json_files)}")

    print(f"Aligned {processed} samples -> {align_dir}\n")
    return str(align_dir)


def step_summary(args):
    """Print a summary of all processed data."""
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    base = Path(args.base_dir)
    for subdir in ["transcriptions", "translations", "alignments"]:
        path = base / subdir
        if path.exists():
            for lang_dir in sorted(path.iterdir()):
                if lang_dir.is_dir():
                    n_files = len(list(lang_dir.glob("*.json")))
                    print(f"  {subdir}/{lang_dir.name}: {n_files} files")

    print()
    print("Next steps:")
    print(f"  1. Synthesize {args.target_lang} speech with TTS (VITS or MMS-TTS)")
    print(f"  2. Encode all audio through Mimi codec (on Kaggle GPU)")
    print(f"  3. Create S2ST training manifest")


def main():
    parser = argparse.ArgumentParser(
        description="Hibiki-Sw data preparation pipeline"
    )
    parser.add_argument("--source", type=str, default="common_voice",
                        choices=["common_voice", "directory"])
    parser.add_argument("--source_lang", type=str, required=True,
                        help="Source language (sw or en)")
    parser.add_argument("--target_lang", type=str, required=True,
                        help="Target language (en or sw)")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base output directory (e.g. /content/drive/MyDrive/hibiki-sw)")
    parser.add_argument("--whisper_model", type=str, default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "transcribe", "translate", "align", "summary"],
                        help="Which step(s) to run")
    args = parser.parse_args()

    start = time.time()

    steps = {
        "transcribe": step_transcribe,
        "translate": step_translate,
        "align": step_align,
        "summary": step_summary,
    }

    if args.step == "all":
        for name, fn in steps.items():
            fn(args)
    else:
        steps[args.step](args)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")


if __name__ == "__main__":
    main()
