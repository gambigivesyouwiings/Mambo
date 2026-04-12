"""End-to-end data preparation pipeline orchestrator.

Runs the full offline data processing chain:
    1. Whisper transcription (GPU)
    2. MADLAD translation (GPU)
    3. Contextual alignment (GPU — uses MADLAD encoder)
    4. TTS synthesis of target-language speech (GPU)
    5. Silence insertion (CPU, integrated into synthesis step)

Designed for Google Colab with a T4 GPU. Saves all outputs to Google Drive
for persistence across Colab sessions.

Usage:
    # Full pipeline for Swahili source audio (Sw->En direction)
    python data/prepare/run_pipeline.py \
        --source common_voice \
        --source_lang sw \
        --target_lang en \
        --dataset_dir /content/cv-corpus-19.0-2024-09-13/sw \
        --base_dir /content/drive/MyDrive/hibiki-sw \
        --whisper_model medium \
        --max_samples 50000 \
        --step all

    # Full pipeline for English source audio (En->Sw direction)
    python data/prepare/run_pipeline.py \
        --source common_voice \
        --source_lang en \
        --target_lang sw \
        --dataset_dir /content/cv-corpus-19.0-2024-09-13/en \
        --base_dir /content/drive/MyDrive/hibiki-sw \
        --vits_model_dir /content/drive/MyDrive/hibiki-sw/vits_sw/hf_model \
        --step all

    # Run only specific steps (for resuming)
    python data/prepare/run_pipeline.py \
        --source common_voice \
        --source_lang sw \
        --target_lang en \
        --dataset_dir /content/cv-corpus-19.0-2024-09-13/sw \
        --base_dir /content/drive/MyDrive/hibiki-sw \
        --step translate
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
    from data.prepare.transcribe_whisper import WhisperTranscriber

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

    if args.source == "kenspeech":
        from data.prepare.transcribe_whisper import process_kenspeech
        n = process_kenspeech(
            lang=args.source_lang,
            output_dir=output_dir,
            transcriber=transcriber,
            max_samples=args.max_samples,
            min_duration=1.0,
            max_duration=30.0,
            resume_from=args.resume_from,
        )
    else:
        from data.prepare.transcribe_whisper import process_common_voice
        n = process_common_voice(
            lang=args.source_lang,
            split=args.cv_split,
            output_dir=output_dir,
            transcriber=transcriber,
            dataset_dir=args.dataset_dir,
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
    print(f"STEP 2: Translate {args.source_lang} -> {args.target_lang}")
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
    print(f"STEP 3: Contextual alignment ({args.source_lang} -> {args.target_lang})")
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


def step_synthesize(args):
    """Step 4: Synthesize target-language speech with TTS."""
    from data.prepare.synthesize_tts import get_tts_backend, synthesize_and_align

    print("=" * 60)
    print(f"STEP 4: Synthesize {args.target_lang} speech with TTS")
    print("=" * 60)

    direction = f"{args.source_lang}2{args.target_lang}"
    translation_dir = os.path.join(args.base_dir, "translations", direction)
    alignment_dir = os.path.join(args.base_dir, "alignments", direction)
    output_dir = os.path.join(args.base_dir, "synthetic_audio", direction)

    # Determine VITS model directory
    vits_dir = args.vits_model_dir
    if vits_dir is None and args.target_lang == "sw":
        # Default location for Stage 0 trained model
        default_vits = os.path.join(args.base_dir, "vits_sw", "hf_model")
        if os.path.exists(default_vits):
            vits_dir = default_vits
            print(f"  Using Stage 0 VITS model: {vits_dir}")
        else:
            print(f"  No fine-tuned VITS found at {default_vits}")
            print(f"  Falling back to pretrained facebook/mms-tts-swh")

    tts = get_tts_backend(
        target_lang=args.target_lang,
        vits_model_dir=vits_dir,
        backend=args.tts_backend,
        device=args.device,
    )

    # Check if alignment data exists
    if os.path.exists(alignment_dir) and any(Path(alignment_dir).glob("*.json")):
        print(f"  Found alignments at {alignment_dir} — running full pipeline")
        n = synthesize_and_align(
            translation_dir=translation_dir,
            alignment_dir=alignment_dir,
            source_audio_dir=os.path.join(args.base_dir, "transcriptions", args.source_lang),
            output_dir=output_dir,
            tts=tts,
            target_lang=args.target_lang,
            whisper_model=args.whisper_model,
            target_sr=24000,
            min_lag=2.0,
            max_samples=args.max_samples,
        )
    else:
        print(f"  No alignments found — running synthesis only (silence insertion skipped)")
        from data.prepare.synthesize_tts import synthesize_from_translations
        n = synthesize_from_translations(
            translation_dir=translation_dir,
            output_dir=output_dir,
            tts=tts,
            target_lang=args.target_lang,
            whisper_model=args.whisper_model,
            target_sr=24000,
            max_samples=args.max_samples,
            resume_from=args.resume_from,
        )

    print(f"Synthesized {n} samples -> {output_dir}\n")
    return output_dir


def step_summary(args):
    """Print a summary of all processed data."""
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    base = Path(args.base_dir)
    for subdir in ["transcriptions", "translations", "alignments", "synthetic_audio"]:
        path = base / subdir
        if path.exists():
            for lang_dir in sorted(path.iterdir()):
                if lang_dir.is_dir():
                    n_json = len(list(lang_dir.glob("*.json")))
                    n_wav = len(list(lang_dir.rglob("*.wav")))
                    parts = []
                    if n_json:
                        parts.append(f"{n_json} json")
                    if n_wav:
                        parts.append(f"{n_wav} wav")
                    print(f"  {subdir}/{lang_dir.name}: {', '.join(parts) if parts else 'empty'}")

    print()
    print("Next steps:")
    print(f"  1. Encode all audio (source + synthesized) through Mimi codec")
    print(f"  2. Create S2ST training manifest")
    print(f"  3. Start Stage 1-2 training on Kaggle")


def main():
    parser = argparse.ArgumentParser(
        description="Hibiki-Sw data preparation pipeline"
    )
    parser.add_argument("--source", type=str, default="common_voice",
                        choices=["common_voice", "kenspeech", "directory"])
    parser.add_argument("--source_lang", type=str, required=True,
                        help="Source language (sw or en)")
    parser.add_argument("--target_lang", type=str, required=True,
                        help="Target language (en or sw)")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base output directory (e.g. /content/drive/MyDrive/hibiki-sw)")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Path to extracted Common Voice language directory, "
                             "e.g. /content/cv-corpus-19.0-2024-09-13/sw")
    parser.add_argument("--cv_split", type=str, default="validated",
                        help="Common Voice split to use (default: validated)")
    parser.add_argument("--whisper_model", type=str, default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume_from", type=int, default=0)

    # TTS options
    parser.add_argument("--vits_model_dir", type=str, default=None,
                        help="Path to fine-tuned VITS model directory "
                             "(auto-detected from base_dir/vits_sw/hf_model if not set)")
    parser.add_argument("--tts_backend", type=str, default="mms",
                        choices=["mms", "coqui"],
                        help="TTS backend for synthesis step")

    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "transcribe", "translate", "align",
                                 "synthesize", "summary"],
                        help="Which step(s) to run")
    args = parser.parse_args()

    start = time.time()

    steps = {
        "transcribe": step_transcribe,
        "translate": step_translate,
        "align": step_align,
        "synthesize": step_synthesize,
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
