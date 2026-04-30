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
        initial_prompt: str = None,
    ) -> Dict:
        """Transcribe from a numpy audio array (for HuggingFace datasets).

        Args:
            audio: 1-D float32 numpy array
            sr: Sample rate of `audio`
            language: Language code for Whisper
            beam_size: Beam search width
            initial_prompt: Optional prompt to guide Whisper's decoding.
                When set to the known ground-truth transcript (e.g., CV `sentence`),
                Whisper produces timestamps that closely follow the known text
                instead of free-form transcribing — useful as a fallback when
                WhisperX forced alignment is unavailable.
        """
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
            initial_prompt=initial_prompt,
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
# Forced alignment (WhisperX)
# ---------------------------------------------------------------------------

def _uniform_timestamps(
    transcript: str, duration: float
) -> List[Tuple[str, float, float]]:
    """Fallback: distribute word timestamps uniformly across audio duration.

    Used when both WhisperX and Whisper+initial_prompt fail. Timestamps are
    inaccurate but at least non-empty, so downstream steps won't crash.
    """
    words = transcript.strip().split()
    if not words:
        return []
    step = duration / len(words)
    return [(w, round(i * step, 3), round((i + 1) * step, 3)) for i, w in enumerate(words)]


class WhisperXAligner:
    """Forced alignment using WhisperX — aligns a *known* transcript to audio.

    Much more accurate than free-form Whisper for low-resource languages like
    Swahili because it never re-transcribes — it only pins the existing text to
    the audio timeline using a wav2vec2 phoneme model.

    Fallback chain (applied automatically):
        1. WhisperX forced alignment  (best)
        2. Uniform timestamp spread   (last resort if whisperx unavailable)

    The caller (process_common_voice) handles the Whisper+initial_prompt middle
    tier when this object is not provided.
    """

    def __init__(self, language: str = "sw", device: str = "cuda"):
        self.language = language
        self.device = device
        self._align_model = None
        self._align_metadata = None
        self._available = False
        self._load()

    def _load(self):
        try:
            import whisperx
            print(f"Loading WhisperX alignment model for '{self.language}'...")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=self.language,
                device=self.device,
            )
            self._available = True
            print("WhisperX alignment model loaded.")
        except ImportError:
            print(
                "  whisperx not installed — forced alignment unavailable.\n"
                "  Install with: pip install whisperx"
            )
        except Exception as e:
            print(
                f"  Could not load WhisperX alignment model for '{self.language}': {e}\n"
                "  Falling back to Whisper+initial_prompt for timestamps."
            )

    @property
    def available(self) -> bool:
        return self._available

    def align(
        self,
        audio: np.ndarray,
        sr: int,
        transcript: str,
    ) -> List[Tuple[str, float, float]]:
        """Align a known transcript to audio at the word level.

        Args:
            audio: 1-D float32 numpy array at any sample rate
            sr: Sample rate of `audio`
            transcript: Ground-truth text (e.g., Common Voice `sentence`)

        Returns:
            List of (word, start_sec, end_sec) tuples, sorted by start time.
            Falls back to uniform distribution if alignment fails.
        """
        if not self._available:
            return _uniform_timestamps(transcript, len(audio) / sr)

        import whisperx

        # WhisperX expects 16 kHz float32
        if sr != 16000:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(16000, sr)
                audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
                sr = 16000
            except Exception:
                pass  # proceed anyway; whisperx may still work

        duration = len(audio) / sr

        try:
            segments = [{"start": 0.0, "end": duration, "text": transcript}]
            aligned = whisperx.align(
                segments,
                self._align_model,
                self._align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            words: List[Tuple[str, float, float]] = []
            for seg in aligned.get("segments", []):
                for w in seg.get("words", []):
                    word_str = w.get("word", "").strip()
                    start = round(float(w.get("start", 0.0)), 3)
                    end = round(float(w.get("end", 0.0)), 3)
                    if word_str:
                        words.append((word_str, start, end))

            if words:
                return words

        except Exception as e:
            pass  # fall through to uniform

        return _uniform_timestamps(transcript, duration)


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_common_voice(
    lang: str,
    split: str,
    output_dir: str,
    transcriber: WhisperTranscriber,
    dataset_dir: str = None,
    max_samples: Optional[int] = None,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    resume_from: int = 0,
    use_forced_alignment: bool = True,
    aligner: "WhisperXAligner" = None,
):
    """Process a locally-downloaded Common Voice dataset.

    When `use_forced_alignment=True` (the default), the CV `sentence` field is
    **always** used as the authoritative transcript text — Whisper's free-form
    output is never used for the text.  Only word-level *timestamps* are
    obtained via one of three fallback tiers:

        Tier 1 — WhisperX forced alignment (best)
            Requires `aligner` to be a loaded ``WhisperXAligner``.
            Uses a wav2vec2 phoneme model to pin the known text to the audio
            timeline.  Produces accurate timestamps even for low-resource
            languages like Swahili.

        Tier 2 — Whisper + initial_prompt (good)
            Used when WhisperX is unavailable.  Passes the CV sentence as
            ``initial_prompt`` so Whisper's decoding stays close to the known
            text, yielding much better timestamps than free-form transcription.

        Tier 3 — Uniform distribution (acceptable fallback)
            Triggered only if Whisper itself crashes.  Timestamps are
            inaccurate but non-empty so downstream steps don't break.

    When `use_forced_alignment=False`, the old behaviour is preserved:
    Whisper transcribes freely and its output text is used.

    Args:
        lang: Language code (e.g., "sw", "en")
        split: TSV split file ("validated", "train", "dev", "test", "other")
        output_dir: Output directory for JSON transcriptions
        transcriber: WhisperTranscriber instance (used for Tier 2 fallback)
        dataset_dir: Path to the extracted Common Voice language directory,
            e.g. "/content/cv-corpus-19.0-2024-09-13/sw".
            This directory should contain clips/ and <split>.tsv files.
        max_samples: Maximum number of samples to process
        min_duration: Skip clips shorter than this (seconds)
        max_duration: Skip clips longer than this (seconds)
        resume_from: Resume from this sample index
        use_forced_alignment: If True, use CV `sentence` as text and forced-
            align for timestamps (recommended for all CV languages).
        aligner: Optional ``WhisperXAligner`` instance.  Required for Tier 1;
            falls back to Tier 2 if None or unavailable.
    """
    from data.prepare.local_cv_loader import CommonVoiceLocal

    if dataset_dir is None:
        raise ValueError(
            "--dataset_dir is required. Point it to the extracted Common Voice "
            "language directory, e.g. /content/cv-corpus-19.0-2024-09-13/sw"
        )

    if use_forced_alignment:
        tier = "WhisperX" if (aligner is not None and aligner.available) else "Whisper+prompt"
        print(f"  Timestamp tier: {tier} (text always from CV sentence)")
    else:
        print("  Timestamp tier: Whisper free-form transcription")

    print(f"Loading Common Voice {lang} (split={split}) from {dataset_dir}...")
    ds = CommonVoiceLocal(dataset_dir=dataset_dir, split=split, load_audio=True)

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

            try:
                sample = ds[i]
            except Exception:
                skipped += 1
                continue

            audio = sample["audio"]
            sr = audio["sampling_rate"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            duration = len(audio_array) / sr

            # Filter by duration
            if duration < min_duration or duration > max_duration:
                skipped += 1
                continue

            cv_sentence = sample.get("sentence", "").strip()

            try:
                if use_forced_alignment:
                    # ---------------------------------------------------------
                    # Tier 1: WhisperX forced alignment
                    # ---------------------------------------------------------
                    if aligner is not None and aligner.available:
                        words = aligner.align(audio_array, sr, cv_sentence)
                        whisper_text = ""
                        language_probability = 1.0
                    else:
                        # -----------------------------------------------------
                        # Tier 2: Whisper guided by initial_prompt=cv_sentence
                        # -----------------------------------------------------
                        whisper_result = transcriber.transcribe_audio_array(
                            audio_array, sr=sr, language=lang,
                            initial_prompt=cv_sentence if cv_sentence else None,
                        )
                        words = whisper_result.get("words", [])
                        whisper_text = whisper_result.get("text", "")
                        language_probability = whisper_result.get("language_probability", 1.0)

                    # Tier 3 fallback: uniform distribution if timestamps empty
                    if not words and cv_sentence:
                        words = _uniform_timestamps(cv_sentence, duration)

                    text = cv_sentence  # CV sentence is ALWAYS the authoritative text

                    result = {
                        "text": text,
                        "words": words,
                        "segments": [],           # segment-level data not needed downstream
                        "language": lang,
                        "language_probability": language_probability,
                        "duration": round(duration, 3),
                        "whisper_text": whisper_text,  # kept for debugging only
                        "forced_alignment": True,
                    }

                else:
                    # ---------------------------------------------------------
                    # Legacy path: free-form Whisper transcription
                    # ---------------------------------------------------------
                    result = transcriber.transcribe_audio_array(
                        audio_array, sr=sr, language=lang
                    )
                    result["forced_alignment"] = False

                # Add metadata common to both paths
                result["sample_idx"] = i
                result["original_sentence"] = cv_sentence
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
                    "original": cv_sentence,
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


def process_kenspeech(
    lang: str,
    output_dir: str,
    transcriber: WhisperTranscriber,
    max_samples: Optional[int] = None,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    resume_from: int = 0,
    local_dir: str = None,
):
    """Process KenSpeech dataset: use existing transcripts, run Whisper for timestamps.

    KenSpeech already has high-quality transcriptions. We use those as the
    authoritative text but still run Whisper to extract word-level timestamps
    needed for the contextual alignment and silence insertion steps.

    Args:
        lang: Language code (should be "sw" for KenSpeech)
        output_dir: Output directory for JSON transcriptions
        transcriber: WhisperTranscriber instance
        max_samples: Maximum number of samples to process
        min_duration: Skip clips shorter than this (seconds)
        max_duration: Skip clips longer than this (seconds)
        resume_from: Resume from this sample index
        local_dir: Path to local KenSpeech directory (e.g. /kaggle/input/kenspeech-sw).
            If None, downloads from HuggingFace.
    """
    from data.prepare.kenspeech_loader import KenSpeechLoader

    print(f"Loading KenSpeech dataset...")
    ds = KenSpeechLoader(load_audio=True, local_dir=local_dir)

    os.makedirs(output_dir, exist_ok=True)

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

            try:
                sample = ds[i]
            except Exception:
                skipped += 1
                continue

            audio = sample["audio"]
            sr = audio["sampling_rate"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            duration = len(audio_array) / sr

            if duration < min_duration or duration > max_duration:
                skipped += 1
                continue

            kenspeech_transcript = sample.get("sentence", "").strip()
            if not kenspeech_transcript:
                skipped += 1
                continue

            try:
                # Run Whisper guided by the known transcript for better timestamps
                whisper_result = transcriber.transcribe_audio_array(
                    audio_array, sr=sr, language=lang,
                    initial_prompt=kenspeech_transcript,
                )

                # Use KenSpeech transcript as the authoritative text,
                # but keep Whisper's word timestamps for alignment
                result = {
                    "text": kenspeech_transcript,
                    "segments": whisper_result.get("segments", []),
                    "words": whisper_result.get("words", []),
                    "language": lang,
                    "language_probability": whisper_result.get("language_probability", 1.0),
                    "duration": round(duration, 3),
                    "sample_idx": i,
                    "original_sentence": kenspeech_transcript,
                    "whisper_text": whisper_result.get("text", ""),
                    "client_id": sample.get("client_id", ""),
                    "path": sample.get("path", ""),
                    "audio_duration": round(duration, 3),
                }

                out_path = os.path.join(output_dir, f"{lang}_{i:07d}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                index_entry = {
                    "idx": i,
                    "file": f"{lang}_{i:07d}.json",
                    "text": kenspeech_transcript,
                    "original": kenspeech_transcript,
                    "duration": result["audio_duration"],
                    "n_words": len(result["words"]),
                }
                idx_f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

                processed += 1

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
                        choices=["common_voice", "kenspeech", "directory"],
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
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Path to extracted Common Voice language directory, "
                             "e.g. /content/cv-corpus-19.0-2024-09-13/sw")
    parser.add_argument("--kenspeech_dir", type=str, default=None,
                        help="Path to local KenSpeech dataset directory "
                             "(e.g. /kaggle/input/kenspeech-sw). "
                             "If omitted, downloads from HuggingFace.")

    # Forced alignment flags (default ON for Common Voice)
    forced_group = parser.add_mutually_exclusive_group()
    forced_group.add_argument(
        "--forced_alignment", dest="forced_alignment", action="store_true",
        default=True,
        help="Use CV sentence as text + forced alignment for timestamps "
             "(default: enabled for common_voice source). "
             "Requires: pip install whisperx",
    )
    forced_group.add_argument(
        "--no_forced_alignment", dest="forced_alignment", action="store_false",
        help="Disable forced alignment — use free-form Whisper transcription instead.",
    )

    args = parser.parse_args()

    transcriber = WhisperTranscriber(
        model_size=args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
    )

    # Instantiate WhisperXAligner for common_voice when forced alignment is on
    aligner = None
    if args.source == "common_voice" and args.forced_alignment:
        aligner = WhisperXAligner(language=args.lang, device=args.device)
        if not aligner.available:
            print(
                "  WhisperX unavailable — falling back to Whisper+initial_prompt "
                "for timestamps (text still taken from CV sentence)."
            )

    if args.source == "common_voice":
        process_common_voice(
            lang=args.lang,
            split=args.split,
            output_dir=args.output_dir,
            transcriber=transcriber,
            dataset_dir=args.dataset_dir,
            max_samples=args.max_samples,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            resume_from=args.resume_from,
            use_forced_alignment=args.forced_alignment,
            aligner=aligner,
        )
    elif args.source == "kenspeech":
        process_kenspeech(
            lang=args.lang,
            output_dir=args.output_dir,
            transcriber=transcriber,
            max_samples=args.max_samples,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            resume_from=args.resume_from,
            local_dir=args.kenspeech_dir,
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
