"""Load a locally-downloaded Mozilla Common Voice dataset.

Common Voice datasets downloaded from Mozilla Data Collective
(https://datacollective.mozillafoundation.org) extract to:

    cv-corpus-XX.X-YYYY-MM-DD/
    └── <lang>/
        ├── clips/            # MP3 audio files
        ├── validated.tsv     # clips verified by ≥2 reviewers
        ├── invalidated.tsv
        ├── train.tsv
        ├── dev.tsv
        ├── test.tsv
        └── other.tsv

TSV columns:
    client_id  path  sentence  up_votes  down_votes  age  gender  accents  variant  locale  segment

This module provides a simple iterator that yields samples matching
the HuggingFace datasets interface so existing code needs minimal changes.

Usage:
    from data.prepare.local_cv_loader import CommonVoiceLocal

    cv = CommonVoiceLocal(
        dataset_dir="/content/cv-corpus-19.0-2024-09-13/sw",
        split="validated",
    )

    for sample in cv:
        audio_array = sample["audio"]["array"]     # np.float32
        sr = sample["audio"]["sampling_rate"]       # int (always 48000 for MP3 → resampled as needed)
        sentence = sample["sentence"]               # str
        client_id = sample["client_id"]             # str
        path = sample["path"]                       # str (filename)
"""

import csv
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np


class CommonVoiceLocal:
    """Iterator over a locally-downloaded Common Voice dataset split.

    Yields dicts with the same keys as HuggingFace's Common Voice:
        {
            "audio": {"array": np.ndarray, "sampling_rate": int},
            "sentence": str,
            "client_id": str,
            "path": str,
            ...
        }

    Args:
        dataset_dir: Path to the language directory, e.g.
            "/content/cv-corpus-19.0-2024-09-13/sw"
        split: TSV split file to load ("validated", "train", "dev", "test", "other").
        clips_subdir: Name of the audio clips subdirectory (default "clips").
        load_audio: If True (default), load and decode audio into numpy arrays.
            Set to False for text-only processing (much faster).
    """

    SAMPLE_RATE = 48000  # Common Voice MP3s are 48kHz

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: str = "validated",
        clips_subdir: str = "clips",
        load_audio: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.clips_dir = self.dataset_dir / clips_subdir
        self.load_audio = load_audio

        tsv_path = self.dataset_dir / f"{split}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {tsv_path}\n"
                f"Available splits: {[f.stem for f in self.dataset_dir.glob('*.tsv')]}"
            )

        # Parse TSV into list of dicts
        self.entries: List[Dict] = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.entries.append(dict(row))

        print(f"CommonVoiceLocal: {len(self.entries)} entries from {tsv_path.name}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample by index."""
        entry = self.entries[idx]
        return self._build_sample(entry)

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over all samples."""
        for entry in self.entries:
            try:
                yield self._build_sample(entry)
            except Exception:
                # Skip samples with missing/corrupt audio
                continue

    def _build_sample(self, entry: Dict) -> Dict:
        """Build a sample dict from a TSV row."""
        filename = entry.get("path", "")
        audio_path = self.clips_dir / filename

        sample = {
            "sentence": entry.get("sentence", ""),
            "client_id": entry.get("client_id", ""),
            "path": filename,
            "up_votes": int(entry.get("up_votes", 0)),
            "down_votes": int(entry.get("down_votes", 0)),
            "age": entry.get("age", ""),
            "gender": entry.get("gender", ""),
            "accents": entry.get("accents", ""),
            "locale": entry.get("locale", ""),
        }

        if self.load_audio:
            audio_array, sr = self._load_audio(audio_path)
            sample["audio"] = {
                "array": audio_array,
                "sampling_rate": sr,
                "path": str(audio_path),
            }
        else:
            sample["audio"] = None

        return sample

    @staticmethod
    def _load_audio(path: Path) -> tuple:
        """Load an audio file and return (numpy_array, sample_rate).

        Tries torchaudio first (fast), falls back to soundfile + scipy.
        """
        path_str = str(path)

        # Try torchaudio (handles MP3 natively)
        try:
            import torchaudio
            waveform, sr = torchaudio.load(path_str)
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.squeeze(0).numpy().astype(np.float32), sr
        except Exception:
            pass

        # Fallback: soundfile (may not handle MP3 without ffmpeg)
        try:
            import soundfile as sf
            audio, sr = sf.read(path_str, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sr
        except Exception:
            pass

        # Last resort: librosa
        try:
            import librosa
            audio, sr = librosa.load(path_str, sr=None, mono=True)
            return audio.astype(np.float32), sr
        except Exception as e:
            raise RuntimeError(f"Cannot load audio: {path_str}") from e

    def text_iterator(self) -> Iterator[str]:
        """Yield only the sentence text (no audio loading). Fast."""
        for entry in self.entries:
            sentence = entry.get("sentence", "").strip()
            if sentence:
                yield sentence

    def get_stats(self) -> Dict:
        """Return basic statistics about the loaded split."""
        sentences = [e.get("sentence", "") for e in self.entries]
        return {
            "split": self.split,
            "total_samples": len(self.entries),
            "unique_speakers": len(set(e.get("client_id", "") for e in self.entries)),
            "avg_sentence_length": np.mean([len(s) for s in sentences]) if sentences else 0,
        }
