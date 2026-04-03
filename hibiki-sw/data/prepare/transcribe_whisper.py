"""Transcribe audio datasets with Whisper and extract word-level timestamps.

Processes Common Voice (or any audio dataset) through faster-whisper to produce:
    1. Transcription text
    2. Word-level timestamps [(word, start_sec, end_sec), ...]
    3. Saves results as JSON files per utterance

Designed for Colab (free T4 GPU) — NOT counted against Kaggle GPU quota.

Usage:
    python data/prepare/transcribe_whisper.py \
        --source common_voice \
        --lang sw \
        --split validated \
        --output_dir /content/drive/MyDrive/hibiki-sw/transcriptions/sw \
        --whisper_model medium \
        --max_samples 50000 \
        --batch_size 16

    # For a local directory of audio files:
    python data/prepare/transcribe_whisper.py \
        --source directory \
        --audio_dir /path/to/audio \
        --lang sw \
        --output_dir /path/to/output \
        --whisper_model medium
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

class WhisperTranscriber:
    """Wrapper around faster-whisper for batch transcription with timestamps."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        print(f"Loading Whisper {self.model_size} on {self.device} ({self.compute_type})...")
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        print("Whisper loaded.")

    def transcribe(
        self,
        audio_path: str,
        language: str = "sw",
        beam_size: int = 5,
        word_timestamps: bool = True,
    ) -> Dict:
        """Transcribe a single audio file.

        Returns:
            dict with keys:
                text: full transcription string
                segments: list of segment dicts
                words: list of (word, start, end) tuples
                language: detected/forced language
                duration: audio duration in seconds
        """
        self._load()

        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )

        segments = []
        words = []
        full_text_parts = []

        for seg in segments_iter:
            seg_dict = {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }

            if seg.words:
                seg_words = []
                for w in seg.words:
                    word_entry = (w.word.strip(), round(w.start, 3), round(w.end, 3))
                    seg_words.append(word_entry)
                    words.append(word_entry)
                seg_dict["words"] = seg_words

            segments.append(seg_dict)
            full_text_parts.append(seg.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "segments": segments,
            "words": words,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 3),
        }

    def transcribe_audio_array(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: str = "sw",
        beam_size: int = 5,
    ) -> Dict:
        """Transcribe from a numpy audio array (for HuggingFace datasets)."""
        self._load()

        # faster-whisper accepts numpy arrays directly
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sr != 16000:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(16000, sr)
                audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
            except ImportError:
                raise ImportError("scipy required for resampling: pip install scipy")

        segments_iter, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=True,
        )

        segments = []
        words = []
        full_text_parts = []

        for seg in segments_iter:
            seg_dict = {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }
            if seg.words:
                seg_words = []
                for w in seg.words:
                    word_entry = (w.word.strip(), round(w.start, 3), round(w.end, 3))
                    seg_words.append(word_entry)
                    words.append(word_entry)
                seg_dict["words"] = seg_words

            segments.append(seg_dict)
            full_text_parts.append(seg.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "segments": segments,
            "words": words,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 3),
        }


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_common_voice(
    lang: str,
    split: str,
    output_dir: str,
    transcriber: WhisperTranscriber,
    max_samples: Optional[int] = None,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    resume_from: int = 0,
):
    """Process Common Voice dataset through Whisper.

    Args:
        lang: Language code (e.g., "sw", "en")
        split: Dataset split ("train", "validated", "test")
        output_dir: Output directory for JSON transcriptions
        transcriber: WhisperTranscriber instance
        max_samples: Maximum number of samples to process
        min_duration: Skip clips shorter than this (seconds)
        max_duration: Skip clips longer than this (seconds)
        resume_from: Resume from this sample index
    """
    from datasets import load_dataset

    # Map split names: Common Voice uses "train", "validation", "test"
    # but also has "validated", "invalidated", "other"
    hf_split = split
    if split == "validated":
        hf_split = "train"  # validated samples are in the train split

    print(f"Loading Common Voice {lang} ({hf_split})...")
    ds = load_dataset(
        "mozilla-foundation/common_voice_16_0",
        lang,
        split=hf_split,
        trust_remote_code=True,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Write metadata index
    index_path = os.path.join(output_dir, "index.jsonl")
    index_mode = "a" if resume_from > 0 else "w"

    processed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    with open(index_path, index_mode, encoding="utf-8") as idx_f:
        for i in tqdm(range(resume_from, len(ds)), initial=resume_from, total=len(ds)):
            if max_samples and processed >= max_samples:
                break

            sample = ds[i]
            audio = sample["audio"]
            sr = audio["sampling_rate"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            duration = len(audio_array) / sr

            # Filter by duration
            if duration < min_duration or duration > max_duration:
                skipped += 1
                continue

            try:
                result = transcriber.transcribe_audio_array(
                    audio_array, sr=sr, language=lang
                )

                # Add metadata
                result["sample_idx"] = i
                result["original_sentence"] = sample.get("sentence", "")
                result["client_id"] = sample.get("client_id", "")
                result["path"] = sample.get("path", "")
                result["audio_duration"] = round(duration, 3)

                # Save individual result
                out_path = os.path.join(output_dir, f"{lang}_{i:07d}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                # Write to index
                index_entry = {
                    "idx": i,
                    "file": f"{lang}_{i:07d}.json",
                    "text": result["text"],
                    "original": result["original_sentence"],
                    "duration": result["audio_duration"],
                    "n_words": len(result["words"]),
                }
                idx_f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

                processed += 1

                # Progress log every 500 samples
                if processed % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    print(f"  Processed: {processed} | Skipped: {skipped} | "
                          f"Errors: {errors} | Rate: {rate:.1f} samples/s | "
                          f"Elapsed: {elapsed/60:.1f}min")

            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  Error on sample {i}: {e}")
                continue

    elapsed = time.time() - start_time
    print(f"\nDone! Processed: {processed} | Skipped: {skipped} | "
          f"Errors: {errors} | Time: {elapsed/60:.1f}min")
    print(f"Output: {output_dir}")
    return processed


def process_audio_directory(
    audio_dir: str,
    lang: str,
    output_dir: str,
    transcriber: WhisperTranscriber,
    extensions: Tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg", ".opus"),
    max_samples: Optional[int] = None,
):
    """Process a directory of audio files through Whisper."""
    audio_dir = Path(audio_dir)
    os.makedirs(output_dir, exist_ok=True)

    files = []
    for ext in extensions:
        files.extend(audio_dir.glob(f"**/*{ext}"))
    files.sort()

    if max_samples:
        files = files[:max_samples]

    print(f"Found {len(files)} audio files in {audio_dir}")

    index_path = os.path.join(output_dir, "index.jsonl")
    processed = 0

    with open(index_path, "w", encoding="utf-8") as idx_f:
        for audio_path in tqdm(files, desc="Transcribing"):
            try:
                result = transcriber.transcribe(str(audio_path), language=lang)

                stem = audio_path.stem
                result["source_file"] = str(audio_path)

                out_path = os.path.join(output_dir, f"{stem}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                index_entry = {
                    "file": f"{stem}.json",
                    "source": str(audio_path),
                    "text": result["text"],
                    "duration": result["duration"],
                    "n_words": len(result["words"]),
                }
                idx_f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
                processed += 1

            except Exception as e:
                print(f"  Error on {audio_path.name}: {e}")
                continue

    print(f"Processed {processed} files -> {output_dir}")
    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Whisper (word-level timestamps)"
    )
    parser.add_argument("--source", type=str, required=True,
                        choices=["common_voice", "directory"],
                        help="Audio source type")
    parser.add_argument("--lang", type=str, required=True,
                        help="Language code (e.g. sw, en)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split for Common Voice")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Audio directory (for --source directory)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for transcription JSONs")
    parser.add_argument("--whisper_model", type=str, default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_type", type=str, default="float16",
                        choices=["float16", "int8", "int8_float16", "float32"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume from this sample index (for interrupted runs)")
    args = parser.parse_args()

    transcriber = WhisperTranscriber(
        model_size=args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
    )

    if args.source == "common_voice":
        process_common_voice(
            lang=args.lang,
            split=args.split,
            output_dir=args.output_dir,
            transcriber=transcriber,
            max_samples=args.max_samples,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            resume_from=args.resume_from,
        )
    elif args.source == "directory":
        if not args.audio_dir:
            parser.error("--audio_dir required for --source directory")
        process_audio_directory(
            audio_dir=args.audio_dir,
            lang=args.lang,
            output_dir=args.output_dir,
            transcriber=transcriber,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
