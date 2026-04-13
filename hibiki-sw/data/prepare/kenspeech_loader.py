"""Load the KenSpeech dataset from HuggingFace.

KenSpeech (https://huggingface.co/datasets/Kencorpus/KenSpeech) provides
high-quality Swahili speech with transcriptions:

    - 5,816 samples (~27.5 hours)
    - 26 speakers (19 female, 7 male)
    - 16kHz mono audio
    - CC-BY-4.0 license

This module provides an iterator matching the CommonVoiceLocal interface
so existing pipeline code works with minimal changes.

Usage:
    from data.prepare.kenspeech_loader import KenSpeechLoader

    ks = KenSpeechLoader(load_audio=True)

    for sample in ks:
        audio_array = sample["audio"]["array"]     # np.float32
        sr = sample["audio"]["sampling_rate"]       # 16000
        sentence = sample["sentence"]               # str (mapped from 'transcript')
        client_id = sample["client_id"]             # str (mapped from 'speaker')
"""

from typing import Dict, Iterator, Optional, Union

import numpy as np


class KenSpeechLoader:
    """Iterator over the KenSpeech dataset from HuggingFace.

    Yields dicts with the same keys as CommonVoiceLocal:
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
        cache_dir: Optional HuggingFace cache directory for offline use.
    """

    SAMPLE_RATE = 16000
    # Known size — used for len() since IterableDataset doesn't support it
    KNOWN_SIZE = 5816

    def __init__(
        self,
        load_audio: bool = True,
        cache_dir: Optional[str] = None,
    ):
        from datasets import load_dataset

        self.load_audio = load_audio

        kwargs: dict = {"streaming": True}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        print("Loading KenSpeech from HuggingFace (streaming)...")
        self.dataset = load_dataset(
            "Kencorpus/KenSpeech",
            split="train",
            **kwargs,
        )
        print("KenSpeechLoader: streaming ready")

    def __len__(self) -> int:
        # IterableDataset has no len(); return known size
        return self.KNOWN_SIZE

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample by index (requires iterating to idx)."""
        for i, row in enumerate(self.dataset):
            if i == idx:
                return self._build_sample(row, idx)
        raise IndexError(f"Index {idx} out of range")

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over all samples."""
        for i, row in enumerate(self.dataset):
            try:
                yield self._build_sample(row, i)
            except Exception:
                continue

    def _build_sample(self, row: Dict, idx: int) -> Dict:
        """Build a sample dict from a HuggingFace dataset row."""
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

    def text_iterator(self) -> Iterator[str]:
        """Yield only the sentence text (no audio loading). Fast."""
        for row in self.dataset:
            text = row.get("transcript", "").strip()
            if text:
                yield text

    def get_stats(self) -> Dict:
        """Return basic statistics about the dataset (single pass)."""
        transcripts = []
        speakers = set()
        for row in self.dataset:
            transcripts.append(row.get("transcript", ""))
            speakers.add(row.get("speaker", ""))
        return {
            "split": "train",
            "total_samples": len(transcripts) or self.KNOWN_SIZE,
            "unique_speakers": len(speakers),
            "avg_sentence_length": (
                np.mean([len(t) for t in transcripts]) if transcripts else 0
            ),
        }
