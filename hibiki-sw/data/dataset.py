"""PyTorch Dataset classes for all Hibiki-Sw training stages.

Mimi codec parameters:
    - Frame rate: 12.5 Hz
    - Codebooks: Q = 8
    - Codebook size: 2048

Token conventions:
    - pad=0, bos=1, eos=2, epad=3
    - Audio token IDs are offset by +4 in the model embeddings, but stored
      raw (0..2047) on disk. The offset is applied by the embedding layers,
      not here.

File layout assumptions:
    Text  (Stage 1): directory of .npy files, each int32 shape (S,)
    Audio (Stage 2): directory of .npy files, each int16 shape (Q, T)
    S2ST  (Stage 3-4): manifest TSV pointing to .npy triplets
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
EPAD_ID = 3

NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 2048
ACOUSTIC_DELAY = 2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_npy_mmap(path: Union[str, Path]) -> np.ndarray:
    """Load a .npy file as a memory-mapped array (read-only)."""
    return np.load(path, mmap_mode="r")


def _pad_or_truncate_1d(seq: np.ndarray, length: int, pad_value: int = PAD_ID) -> np.ndarray:
    """Pad or truncate a 1-D array to exactly *length*."""
    if len(seq) >= length:
        return seq[:length]
    padded = np.full(length, pad_value, dtype=seq.dtype)
    padded[: len(seq)] = seq
    return padded


def _pad_or_truncate_2d(arr: np.ndarray, length: int, pad_value: int = PAD_ID) -> np.ndarray:
    """Pad or truncate a 2-D array (Q, T) along the time axis."""
    Q, T = arr.shape
    if T >= length:
        return arr[:, :length]
    padded = np.full((Q, length), pad_value, dtype=arr.dtype)
    padded[:, :T] = arr
    return padded


def apply_acoustic_delay_np(tokens: np.ndarray, delay: int = ACOUSTIC_DELAY) -> np.ndarray:
    """Apply acoustic delay pattern to (Q, T) token array.

    Semantic codebook (q=0) is unchanged. Acoustic codebooks (q>=1) are
    shifted right by *delay* steps and zero-padded at the front.
    """
    Q, T = tokens.shape
    out = tokens.copy()
    if Q > 1 and delay > 0:
        out[1:, delay:] = tokens[1:, :-delay]
        out[1:, :delay] = PAD_ID
    return out


# ---------------------------------------------------------------------------
# Stage 1: Text Pretraining / Adaptation
# ---------------------------------------------------------------------------


class TextDataset(Dataset):
    """Dataset for Stage 1 text pretraining.

    Each sample is a pre-tokenized text sequence stored as a 1-D int32 .npy
    file. Returns (input_ids, labels) for next-token prediction where
    labels are the input shifted left by one position.

    Args:
        data_dir: directory containing .npy token files.
        seq_length: maximum sequence length (tokens are truncated or padded).
    """

    def __init__(self, data_dir: Union[str, Path], seq_length: int = 1024):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.files: List[Path] = sorted(self.data_dir.glob("*.npy"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = _load_npy_mmap(self.files[idx]).astype(np.int64)

        # Ensure room for the BOS prefix and EOS suffix
        # Sequence layout: [BOS, tok_0, tok_1, ..., tok_{N-1}, EOS, PAD...]
        # We need seq_length + 1 tokens so that input and label are both
        # seq_length long after the shift.
        max_content = self.seq_length - 1  # reserve 1 for BOS
        tokens = tokens[:max_content]

        seq = np.full(self.seq_length + 1, PAD_ID, dtype=np.int64)
        seq[0] = BOS_ID
        seq[1 : 1 + len(tokens)] = tokens
        if 1 + len(tokens) < self.seq_length + 1:
            seq[1 + len(tokens)] = EOS_ID

        input_ids = torch.from_numpy(seq[:-1].copy())   # (seq_length,)
        labels = torch.from_numpy(seq[1:].copy())        # (seq_length,)

        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Stage 2: Audio Pretraining (monolingual)
# ---------------------------------------------------------------------------


class AudioPretrainDataset(Dataset):
    """Dataset for Stage 2 monolingual audio pretraining.

    Each sample is a pre-encoded Mimi token array of shape (Q, T) stored as
    a .npy file (int16). At load time we randomly crop to a fixed number of
    frames, flatten the codebooks into a single stream, and return
    (input_ids, labels) for next-token prediction.

    The single-stream interleaving order follows Mimi convention:
        t=0: q0 q1 ... q7, t=1: q0 q1 ... q7, ...
    giving a 1-D sequence of length Q * T_crop.

    Args:
        data_dir: directory containing .npy token files.
        max_frames: number of audio frames to crop to (time steps).
    """

    def __init__(self, data_dir: Union[str, Path], max_frames: int = 250):
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.files: List[Path] = sorted(self.data_dir.glob("*.npy"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = _load_npy_mmap(self.files[idx]).astype(np.int64)  # (Q, T)
        Q, T = tokens.shape

        # Random crop
        if T > self.max_frames:
            start = random.randint(0, T - self.max_frames)
            tokens = tokens[:, start : start + self.max_frames]
        else:
            tokens = _pad_or_truncate_2d(tokens, self.max_frames)

        # Flatten to single stream: interleave codebooks per timestep
        # Result shape: (Q * max_frames,)
        flat = tokens.T.reshape(-1)  # (T, Q) -> (T*Q,)

        # Prepend BOS, append EOS
        seq_len = len(flat)
        seq = np.full(seq_len + 2, PAD_ID, dtype=np.int64)
        seq[0] = BOS_ID
        seq[1 : seq_len + 1] = flat
        seq[seq_len + 1] = EOS_ID

        input_ids = torch.from_numpy(seq[:-1].copy())
        labels = torch.from_numpy(seq[1:].copy())

        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Stages 3-4: Speech-to-Speech Translation
# ---------------------------------------------------------------------------


class S2STDataset(Dataset):
    """Dataset for Stages 3 and 4 speech translation training.

    Expects a manifest TSV file with columns:
        source_audio_path  target_audio_path  text_path  voice_category

    Audio files are .npy of shape (Q, T). Text files are .npy of shape (T,)
    containing aligned text token IDs (one per audio frame, with epad=3 for
    frames that have no new text token).

    The acoustic delay pattern is applied to both source and target audio
    tokens inside this dataset so the model receives ready-to-use delayed
    sequences.

    Args:
        manifest_path: path to TSV manifest.
        max_frames: maximum number of audio frames.
        acoustic_delay: delay (in frames) for acoustic codebooks.
        noise_augmentation: if True, randomly zero out some source audio
            tokens to improve robustness.
        noise_prob: probability of zeroing a source frame when noise
            augmentation is enabled.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        max_frames: int = 250,
        acoustic_delay: int = ACOUSTIC_DELAY,
        noise_augmentation: bool = False,
        noise_prob: float = 0.05,
    ):
        self.manifest_path = Path(manifest_path)
        self.max_frames = max_frames
        self.acoustic_delay = acoustic_delay
        self.noise_augmentation = noise_augmentation
        self.noise_prob = noise_prob

        self.entries: List[Tuple[str, str, str, int]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                src_audio, tgt_audio, text, voice_cat = parts[:4]
                self.entries.append((src_audio, tgt_audio, text, int(voice_cat)))

        if len(self.entries) == 0:
            raise FileNotFoundError(
                f"No valid entries found in manifest {self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_path, tgt_path, txt_path, voice_cat = self.entries[idx]

        # Load arrays via memory map
        src_audio = _load_npy_mmap(src_path).astype(np.int64)  # (Q, T_src)
        tgt_audio = _load_npy_mmap(tgt_path).astype(np.int64)  # (Q, T_tgt)
        text = _load_npy_mmap(txt_path).astype(np.int64)       # (T_tgt,)

        # Truncate / pad to max_frames
        src_audio = _pad_or_truncate_2d(src_audio, self.max_frames)
        tgt_audio = _pad_or_truncate_2d(tgt_audio, self.max_frames)
        text = _pad_or_truncate_1d(text, self.max_frames, pad_value=PAD_ID)

        # Apply acoustic delay to both source and target
        src_audio = apply_acoustic_delay_np(src_audio, self.acoustic_delay)
        tgt_audio = apply_acoustic_delay_np(tgt_audio, self.acoustic_delay)

        # Noise augmentation: randomly replace source frames with PAD
        if self.noise_augmentation:
            mask = np.random.rand(self.max_frames) < self.noise_prob
            src_audio[:, mask] = PAD_ID

        return {
            "source_audio": torch.from_numpy(src_audio.copy()),   # (Q, T)
            "target_audio": torch.from_numpy(tgt_audio.copy()),   # (Q, T)
            "text": torch.from_numpy(text.copy()),                 # (T,)
            "voice_category": torch.tensor(voice_cat, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def collate_text(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate for TextDataset. Pads to longest sequence in batch."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
    return {"input_ids": input_ids, "labels": labels}


def collate_audio(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate for AudioPretrainDataset. Pads to longest sequence in batch."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
    return {"input_ids": input_ids, "labels": labels}


def collate_s2st(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate for S2STDataset. Pads audio (Q, T) and text (T) to max T."""
    max_t = max(b["source_audio"].shape[1] for b in batch)
    B = len(batch)
    Q = batch[0]["source_audio"].shape[0]

    source_audio = torch.full((B, Q, max_t), PAD_ID, dtype=torch.long)
    target_audio = torch.full((B, Q, max_t), PAD_ID, dtype=torch.long)
    text = torch.full((B, max_t), PAD_ID, dtype=torch.long)
    voice_category = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        T = b["source_audio"].shape[1]
        source_audio[i, :, :T] = b["source_audio"]
        target_audio[i, :, :T] = b["target_audio"]
        text[i, :T] = b["text"]
        voice_category[i] = b["voice_category"]

    return {
        "source_audio": source_audio,
        "target_audio": target_audio,
        "text": text,
        "voice_category": voice_category,
    }
