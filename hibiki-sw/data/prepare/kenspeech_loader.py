"""Load the KenSpeech dataset from local disk or HuggingFace.

KenSpeech (https://huggingface.co/datasets/Kencorpus/KenSpeech) provides
high-quality Swahili speech with transcriptions:

    - 5,816 samples (~27.5 hours)
    - 26 speakers (19 female, 7 male)
    - 16kHz mono audio
    - CC-BY-4.0 license

Preferred usage (local Kaggle dataset — no download):
    ks = KenSpeechLoader(local_dir='/kaggle/input/kenspeech-sw')

Fallback (downloads from HuggingFace):
    ks = KenSpeechLoader()
"""

import csv
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np


class KenSpeechLoader:
    """Iterator over the KenSpeech dataset.

    Loads from a local WAV + CSV directory when ``local_dir`` is provided
    (fast, no network). Falls back to HuggingFace when ``local_dir`` is None.

    Yields dicts matching the CommonVoiceLocal interface:
        {
            "audio": {"array": np.ndarray, "sampling_rate": int},
            "sentence": str,
            "client_id": str,
            "path": str,
            ...
        }

    Args:
        load_audio: If True (default), include decoded audio arrays.
            Set to False for text-only processing (much faster).
        local_dir: Path to local KenSpeech directory containing
            ``metadata.csv`` and ``audio/`` subfolder. On Kaggle this is
            ``/kaggle/input/kenspeech-sw``.
        cache_dir: HuggingFace cache dir (only used when local_dir is None).
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        load_audio: bool = True,
        local_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.load_audio = load_audio
        self._local_dir = Path(local_dir) if local_dir else None
        self._metadata = []

        if self._local_dir:
            self._load_local()
        else:
            self._load_hf(cache_dir)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_local(self):
        csv_path = self._local_dir / "metadata.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"metadata.csv not found in {self._local_dir}. "
                "Run hibiki-sw/scripts/download_kenspeech_kaggle.py first."
            )
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._metadata = list(reader)
        print(f"KenSpeechLoader: {len(self._metadata)} entries loaded from {self._local_dir}")

    def _load_hf(self, cache_dir):
        from datasets import load_dataset
        kwargs: dict = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        print("Loading KenSpeech from HuggingFace (no local_dir provided)...")
        self._hf_dataset = load_dataset("Kencorpus/KenSpeech", split="train", **kwargs)
        print(f"KenSpeechLoader: {len(self._hf_dataset)} entries loaded")

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._local_dir:
            return len(self._metadata)
        return len(self._hf_dataset)

    def __getitem__(self, idx: int) -> Dict:
        if self._local_dir:
            return self._build_local_sample(self._metadata[idx], idx)
        row = self._hf_dataset[idx]
        return self._build_hf_sample(row, idx)

    def __iter__(self) -> Iterator[Dict]:
        for i in range(len(self)):
            try:
                yield self[i]
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Sample builders
    # ------------------------------------------------------------------

    def _build_local_sample(self, row: dict, idx: int) -> Dict:
        sample = {
            "sentence": row.get("transcript", ""),
            "client_id": row.get("speaker", ""),
            "path": row.get("path", f"kenspeech_{idx:05d}"),
            "up_votes": 0,
            "down_votes": 0,
            "age": "",
            "gender": row.get("gender", ""),
            "accents": "",
            "locale": "sw",
        }
        if self.load_audio:
            wav_path = self._local_dir / row["path"]
            import soundfile as sf
            array, sr = sf.read(str(wav_path), dtype="float32")
            sample["audio"] = {
                "array": array,
                "sampling_rate": int(sr),
                "path": str(wav_path),
            }
        else:
            sample["audio"] = None
        return sample

    def _build_hf_sample(self, row: dict, idx: int) -> Dict:
        sample = {
            "sentence": row.get("transcript", ""),
            "client_id": row.get("speaker", ""),
            "path": f"kenspeech_{idx:05d}",
            "up_votes": 0,
            "down_votes": 0,
            "age": "",
            "gender": row.get("gender", ""),
            "accents": "",
            "locale": "sw",
        }
        if self.load_audio:
            audio_data = row.get("audio", {})
            array = np.array(audio_data.get("array", []), dtype=np.float32)
            sr = audio_data.get("sampling_rate", self.SAMPLE_RATE)
            sample["audio"] = {
                "array": array,
                "sampling_rate": sr,
                "path": f"kenspeech_{idx:05d}",
            }
        else:
            sample["audio"] = None
        return sample

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def text_iterator(self) -> Iterator[str]:
        """Yield only transcript text (no audio). Fast."""
        if self._local_dir:
            for row in self._metadata:
                text = row.get("transcript", "").strip()
                if text:
                    yield text
        else:
            for row in self._hf_dataset:
                text = row.get("transcript", "").strip()
                if text:
                    yield text

    def get_stats(self) -> Dict:
        """Return basic statistics (single pass)."""
        transcripts = []
        speakers = set()
        if self._local_dir:
            for row in self._metadata:
                transcripts.append(row.get("transcript", ""))
                speakers.add(row.get("speaker", ""))
        else:
            for row in self._hf_dataset:
                transcripts.append(row.get("transcript", ""))
                speakers.add(row.get("speaker", ""))
        return {
            "split": "train",
            "total_samples": len(transcripts),
            "unique_speakers": len(speakers),
            "avg_sentence_length": (
                np.mean([len(t) for t in transcripts]) if transcripts else 0
            ),
        }
