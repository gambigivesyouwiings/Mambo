"""Synthesize target-language speech from translations using TTS.

This is the critical bridge step that creates the synthetic parallel speech
corpus. It takes translated text (MADLAD output) and produces target-language
audio, which is then processed through silence insertion and Mimi encoding.

Two directions:
- En->Sw: Uses VITS fine-tuned on Common Voice Swahili (Stage 0 output)
- Sw->En: Uses pretrained English TTS (facebook/mms-tts-eng or coqui/tts)

The synthesized audio is also run through Whisper to get word-level timestamps
needed by the silence insertion step.

Usage:
    # Synthesize Swahili speech from En->Sw translations
    python data/prepare/synthesize_tts.py \
        --translation_dir /content/drive/MyDrive/hibiki-sw/translations/en2sw \
        --output_dir /content/drive/MyDrive/hibiki-sw/synthetic_audio/sw \
        --target_lang sw \
        --vits_model_dir /content/drive/MyDrive/hibiki-sw/vits_sw/hf_model \
        --whisper_model medium

    # Synthesize English speech from Sw->En translations (uses pretrained)
    python data/prepare/synthesize_tts.py \
        --translation_dir /content/drive/MyDrive/hibiki-sw/translations/sw2en \
        --output_dir /content/drive/MyDrive/hibiki-sw/synthetic_audio/en \
        --target_lang en \
        --whisper_model medium
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# TTS Backends
# ---------------------------------------------------------------------------

class TTSBackend:
    """Abstract TTS interface."""

    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text.

        Returns:
            (waveform_np, sample_rate) — waveform is float32, mono
        """
        raise NotImplementedError


class MMS_TTS(TTSBackend):
    """Meta MMS-TTS backend (supports many languages, including Swahili)."""

    def __init__(self, lang: str = "swh", model_dir: Optional[str] = None, device: str = "cuda"):
        from transformers import VitsModel, VitsTokenizer

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if model_dir and os.path.exists(model_dir):
            # Load fine-tuned model from Stage 0
            print(f"Loading fine-tuned VITS from {model_dir}")
            self.tokenizer = VitsTokenizer.from_pretrained(model_dir)
            self.model = VitsModel.from_pretrained(model_dir).to(self.device)
        else:
            # Load pretrained MMS-TTS
            model_name = f"facebook/mms-tts-{lang}"
            print(f"Loading pretrained {model_name}")
            self.tokenizer = VitsTokenizer.from_pretrained(model_name)
            self.model = VitsModel.from_pretrained(model_name).to(self.device)

        self.model.eval()
        self.sample_rate = self.model.config.sampling_rate  # typically 16000

    @torch.no_grad()
    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> Tuple[np.ndarray, int]:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            output = self.model(**inputs)

        waveform = output.waveform.squeeze().cpu().numpy()
        return waveform.astype(np.float32), self.sample_rate


class CoquiTTS(TTSBackend):
    """Coqui TTS backend (richer voice quality, multi-speaker)."""

    def __init__(self, model_path: str = None, lang: str = "sw", device: str = "cuda"):
        try:
            from TTS.api import TTS as CoquiTTSAPI
        except ImportError:
            raise ImportError("Coqui TTS required: pip install TTS")

        if model_path and os.path.exists(model_path):
            print(f"Loading Coqui TTS from {model_path}")
            self.tts = CoquiTTSAPI(model_path=model_path, gpu=(device == "cuda"))
        else:
            # Use built-in multilingual model
            print("Loading Coqui TTS default multilingual model")
            self.tts = CoquiTTSAPI("tts_models/multilingual/multi-dataset/xtts_v2",
                                   gpu=(device == "cuda"))

        self.sample_rate = self.tts.synthesizer.output_sample_rate if hasattr(self.tts, "synthesizer") else 22050

    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> Tuple[np.ndarray, int]:
        wav = self.tts.tts(text=text)
        waveform = np.array(wav, dtype=np.float32)
        return waveform, self.sample_rate


def get_tts_backend(
    target_lang: str,
    vits_model_dir: Optional[str] = None,
    backend: str = "mms",
    device: str = "cuda",
) -> TTSBackend:
    """Factory: get the right TTS backend for the target language."""

    # Language code mapping for MMS-TTS
    MMS_LANG_MAP = {
        "sw": "swh",   # Swahili
        "en": "eng",   # English
    }

    if backend == "mms":
        mms_lang = MMS_LANG_MAP.get(target_lang, target_lang)
        return MMS_TTS(lang=mms_lang, model_dir=vits_model_dir, device=device)
    elif backend == "coqui":
        return CoquiTTS(model_path=vits_model_dir, lang=target_lang, device=device)
    else:
        raise ValueError(f"Unknown TTS backend: {backend}")


# ---------------------------------------------------------------------------
# Synthesis pipeline
# ---------------------------------------------------------------------------

def synthesize_from_translations(
    translation_dir: str,
    output_dir: str,
    tts: TTSBackend,
    target_lang: str,
    whisper_model: str = "medium",
    target_sr: int = 24000,
    max_samples: Optional[int] = None,
    resume_from: int = 0,
    get_timestamps: bool = True,
) -> int:
    """Synthesize speech for all translation JSONs in a directory.

    For each translation JSON:
    1. Extract translated text
    2. Synthesize speech with TTS
    3. Resample to 24kHz (Mimi codec rate)
    4. Optionally get word-level timestamps with Whisper
    5. Save WAV + metadata JSON

    Args:
        translation_dir: Directory containing translation JSON files from MADLAD
        output_dir: Output directory for synthesized audio + metadata
        tts: TTS backend instance
        target_lang: Target language code
        whisper_model: Whisper model for forced alignment timestamps
        target_sr: Target sample rate (24000 for Mimi)
        max_samples: Maximum samples to synthesize
        resume_from: Resume from this file index
        get_timestamps: Whether to run Whisper for word timestamps

    Returns:
        Number of successfully synthesized samples
    """
    trans_dir = Path(translation_dir)
    json_files = sorted(
        f for f in trans_dir.glob("*.json")
        if f.name != "index.jsonl"
    )

    if not json_files:
        print(f"No translation JSON files found in {translation_dir}")
        return 0

    if max_samples:
        json_files = json_files[:max_samples]

    wav_dir = os.path.join(output_dir, "wavs")
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # Optionally load Whisper for timestamps
    transcriber = None
    if get_timestamps:
        try:
            from data.prepare.transcribe_whisper import WhisperTranscriber
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            transcriber = WhisperTranscriber(
                model_size=whisper_model, device=device, compute_type=compute_type,
            )
            print(f"Loaded Whisper ({whisper_model}) for word timestamps")
        except Exception as e:
            print(f"Warning: Could not load Whisper for timestamps: {e}")
            print("Synthesized audio will not have word-level timestamps.")

    count = 0
    failed = 0

    for idx, json_path in enumerate(tqdm(json_files, desc=f"Synthesizing {target_lang}")):
        if idx < resume_from:
            continue

        # Load translation
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            failed += 1
            continue

        translated_text = data.get("translated_text", "").strip()
        if not translated_text:
            failed += 1
            continue

        # Synthesize
        try:
            waveform, sr = tts.synthesize(translated_text)
        except Exception as e:
            if idx % 500 == 0:
                print(f"  Synthesis failed for {json_path.name}: {e}")
            failed += 1
            continue

        # Validate output
        if waveform is None or len(waveform) < sr * 0.1:  # < 100ms
            failed += 1
            continue

        # Resample to target_sr (24kHz for Mimi)
        waveform_t = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        if sr != target_sr:
            waveform_t = torchaudio.functional.resample(waveform_t, sr, target_sr)

        # Normalize
        peak = waveform_t.abs().max()
        if peak > 0:
            waveform_t = waveform_t / peak * 0.95

        # Save WAV
        stem = json_path.stem
        wav_path = os.path.join(wav_dir, f"{stem}.wav")
        torchaudio.save(wav_path, waveform_t, target_sr)

        # Get word timestamps via Whisper (for silence insertion)
        word_timestamps = []
        if transcriber is not None:
            try:
                audio_np = waveform_t.squeeze(0).numpy()
                result = transcriber.transcribe(audio_np, language=target_lang)
                word_timestamps = result.get("words", [])
            except Exception:
                pass  # timestamps are optional; silence insertion can fall back

        # Build metadata
        duration = waveform_t.shape[1] / target_sr
        meta = {
            "source_file": data.get("source_file", stem),
            "source_text": data.get("source_text", ""),
            "translated_text": translated_text,
            "wav_path": wav_path,
            "duration": duration,
            "sample_rate": target_sr,
            "target_lang": target_lang,
            "word_timestamps": word_timestamps,
            # Carry forward source info for alignment
            "source_words": data.get("source_words", []),
            "source_word_timestamps": data.get("word_timestamps", []),
            "sample_idx": data.get("sample_idx", idx),
        }

        meta_path = os.path.join(meta_dir, f"{stem}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        count += 1

        if count % 500 == 0:
            print(f"  Synthesized: {count}, Failed: {failed}")

    print(f"\nDone! Synthesized: {count}, Failed: {failed}")
    print(f"  WAVs: {wav_dir}")
    print(f"  Metadata: {meta_dir}")

    # Write index for downstream scripts
    index_path = os.path.join(output_dir, "index.jsonl")
    with open(index_path, "w", encoding="utf-8") as f:
        for meta_file in sorted(Path(meta_dir).glob("*.json")):
            with open(meta_file, "r") as mf:
                f.write(mf.read().replace("\n", " ").strip() + "\n")
    print(f"  Index: {index_path}")

    return count


# ---------------------------------------------------------------------------
# Full pipeline: Synthesize + Align + Silence-Insert
# ---------------------------------------------------------------------------

def synthesize_and_align(
    translation_dir: str,
    alignment_dir: str,
    source_audio_dir: str,
    output_dir: str,
    tts: TTSBackend,
    target_lang: str,
    whisper_model: str = "medium",
    target_sr: int = 24000,
    min_lag: float = 2.0,
    max_samples: Optional[int] = None,
) -> int:
    """Full pipeline: synthesize, align, and insert silences.

    This combines:
    1. TTS synthesis → target audio
    2. Contextual alignment data (pre-computed) → alignment pairs
    3. Silence insertion → causally-aligned target audio

    The output is ready for Mimi encoding.
    """
    from data.silence_insertion import insert_silences, SAMPLE_RATE

    # Step 1: Synthesize
    print("\n=== Step 1: TTS Synthesis ===")
    synth_dir = os.path.join(output_dir, "raw_synthesis")
    n_synth = synthesize_from_translations(
        translation_dir=translation_dir,
        output_dir=synth_dir,
        tts=tts,
        target_lang=target_lang,
        whisper_model=whisper_model,
        target_sr=target_sr,
        max_samples=max_samples,
    )

    if n_synth == 0:
        print("No samples synthesized. Aborting.")
        return 0

    # Step 2: Apply silence insertion using pre-computed alignments
    print("\n=== Step 2: Silence Insertion ===")
    aligned_dir = os.path.join(output_dir, "aligned_audio")
    os.makedirs(aligned_dir, exist_ok=True)

    align_path = Path(alignment_dir)
    synth_meta_dir = Path(synth_dir) / "meta"
    synth_wav_dir = Path(synth_dir) / "wavs"

    count = 0
    skipped = 0

    for meta_file in tqdm(sorted(synth_meta_dir.glob("*.json")), desc="Inserting silences"):
        with open(meta_file, "r") as f:
            meta = json.load(f)

        stem = meta_file.stem

        # Find corresponding alignment file
        align_file = align_path / f"{stem}_aligned.json"
        if not align_file.exists():
            # Try alternate naming patterns
            align_file = align_path / f"{stem}.json"
        if not align_file.exists():
            skipped += 1
            continue

        with open(align_file, "r") as f:
            align_data = json.load(f)

        # Load synthesized audio
        wav_path = synth_wav_dir / f"{stem}.wav"
        if not wav_path.exists():
            skipped += 1
            continue

        try:
            target_waveform, sr = torchaudio.load(str(wav_path))
            target_audio = target_waveform.squeeze(0).numpy()

            # Get timestamps
            target_timestamps = [
                (w.get("word", w.get("text", "")),
                 w.get("start", 0.0),
                 w.get("end", 0.0))
                for w in meta.get("word_timestamps", [])
            ]

            source_timestamps = [
                (w.get("word", w.get("text", "")),
                 w.get("start", 0.0),
                 w.get("end", 0.0))
                for w in meta.get("source_word_timestamps", [])
            ]

            alignment = align_data.get("alignment", [])

            if not target_timestamps or not source_timestamps or not alignment:
                # No timestamps — save raw synthesized audio without silence insertion
                out_path = os.path.join(aligned_dir, f"{stem}.wav")
                torchaudio.save(out_path, target_waveform, sr)
                count += 1
                continue

            # Convert alignment to list of tuples
            alignment_tuples = [(a[0], a[1]) for a in alignment]

            # Insert silences
            result = insert_silences(
                target_audio=target_audio,
                target_timestamps=target_timestamps,
                source_timestamps=source_timestamps,
                alignment=alignment_tuples,
                min_lag=min_lag,
                sample_rate=sr,
            )

            # Save aligned audio
            out_path = os.path.join(aligned_dir, f"{stem}.wav")
            aligned_t = torch.tensor(result.audio, dtype=torch.float32).unsqueeze(0)
            torchaudio.save(out_path, aligned_t, sr)

            # Save updated metadata
            aligned_meta = {
                **meta,
                "aligned_wav_path": out_path,
                "aligned_duration": result.modified_duration,
                "original_synth_duration": result.original_duration,
                "num_silences_inserted": len(result.inserted_silences),
                "modified_timestamps": [
                    {"word": w, "start": s, "end": e}
                    for w, s, e in result.modified_timestamps
                ],
            }

            meta_out = os.path.join(aligned_dir, f"{stem}.json")
            with open(meta_out, "w", encoding="utf-8") as f:
                json.dump(aligned_meta, f, ensure_ascii=False, indent=2)

            count += 1

        except Exception as e:
            if skipped % 100 == 0:
                print(f"  Error processing {stem}: {e}")
            skipped += 1
            continue

    print(f"Silence insertion complete: {count} processed, {skipped} skipped")
    print(f"Output: {aligned_dir}")

    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synthesize target-language speech from translated text"
    )

    # I/O
    parser.add_argument("--translation_dir", type=str, required=True,
                        help="Directory with MADLAD translation JSONs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for synthesized audio")
    parser.add_argument("--target_lang", type=str, required=True,
                        choices=["en", "sw"],
                        help="Target language to synthesize")

    # TTS config
    parser.add_argument("--vits_model_dir", type=str, default=None,
                        help="Path to fine-tuned VITS model directory "
                             "(required for sw, optional for en)")
    parser.add_argument("--tts_backend", type=str, default="mms",
                        choices=["mms", "coqui"],
                        help="TTS backend to use")
    parser.add_argument("--device", type=str, default="cuda")

    # Whisper for timestamps
    parser.add_argument("--whisper_model", type=str, default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"])
    parser.add_argument("--no_timestamps", action="store_true",
                        help="Skip Whisper timestamp extraction")

    # Alignment + silence insertion
    parser.add_argument("--alignment_dir", type=str, default=None,
                        help="Pre-computed alignment directory (for full pipeline)")
    parser.add_argument("--source_audio_dir", type=str, default=None,
                        help="Source audio directory (for full pipeline)")
    parser.add_argument("--min_lag", type=float, default=2.0,
                        help="Minimum lag for silence insertion (seconds)")

    # Processing
    parser.add_argument("--target_sr", type=int, default=24000,
                        help="Target sample rate (24000 for Mimi codec)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume_from", type=int, default=0)

    args = parser.parse_args()

    # Initialize TTS
    tts = get_tts_backend(
        target_lang=args.target_lang,
        vits_model_dir=args.vits_model_dir,
        backend=args.tts_backend,
        device=args.device,
    )

    # Choose mode: simple synthesis vs full pipeline
    if args.alignment_dir:
        # Full pipeline: synthesize + silence insertion
        synthesize_and_align(
            translation_dir=args.translation_dir,
            alignment_dir=args.alignment_dir,
            source_audio_dir=args.source_audio_dir or args.translation_dir,
            output_dir=args.output_dir,
            tts=tts,
            target_lang=args.target_lang,
            whisper_model=args.whisper_model,
            target_sr=args.target_sr,
            min_lag=args.min_lag,
            max_samples=args.max_samples,
        )
    else:
        # Simple synthesis only
        synthesize_from_translations(
            translation_dir=args.translation_dir,
            output_dir=args.output_dir,
            tts=tts,
            target_lang=args.target_lang,
            whisper_model=args.whisper_model,
            target_sr=args.target_sr,
            max_samples=args.max_samples,
            resume_from=args.resume_from,
            get_timestamps=not args.no_timestamps,
        )


if __name__ == "__main__":
    main()
