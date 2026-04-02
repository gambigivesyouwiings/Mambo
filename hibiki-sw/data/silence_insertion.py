"""Silence insertion for causal alignment enforcement (Hibiki Section 3.2.2).

Given target audio, word-level timestamps for both source and target, and
the contextual alignment mapping, this module inserts silence segments into
the target waveform so that every target word is heard *after* its aligned
source word (plus a minimum lag).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default sample rate used by the Hibiki codec / training pipeline.
SAMPLE_RATE = 24_000


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

WordTimestamp = Tuple[str, float, float]  # (word, start_sec, end_sec)


@dataclass
class SilenceInsertionResult:
    """Container for the output of :func:`insert_silences`."""

    audio: np.ndarray
    """Modified waveform with silences inserted (1-D float32)."""

    original_duration: float
    """Duration of the original target audio in seconds."""

    modified_duration: float
    """Duration of the modified audio in seconds."""

    inserted_silences: List[Dict]
    """Per-insertion metadata: target word index, insertion point, duration."""

    modified_timestamps: List[WordTimestamp]
    """Target word timestamps after silence insertion."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seconds_to_samples(seconds: float, sr: int = SAMPLE_RATE) -> int:
    return int(round(seconds * sr))


def _samples_to_seconds(samples: int, sr: int = SAMPLE_RATE) -> float:
    return samples / sr


def _make_silence(duration_sec: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a zero-valued float32 array representing silence."""
    n = max(0, _seconds_to_samples(duration_sec, sr))
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def insert_silences(
    target_audio: np.ndarray,
    target_timestamps: Sequence[WordTimestamp],
    source_timestamps: Sequence[WordTimestamp],
    alignment: Sequence[Tuple[int, int]],
    min_lag: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
) -> SilenceInsertionResult:
    """Insert silence into *target_audio* to enforce causal alignment.

    For each target word *j* whose aligned source word index is *i*:
      - Required onset = source_timestamps[i].start + min_lag
      - If target_timestamps[j].start < required onset, insert silence of
        length (required_onset - current_onset) right before the target word.

    Silences are inserted left-to-right; each insertion shifts all subsequent
    timestamps forward by the inserted amount.

    Parameters
    ----------
    target_audio : np.ndarray
        1-D float waveform at *sample_rate* Hz.
    target_timestamps : sequence of (word, start, end)
        Word-level timestamps in the **original** target audio.
    source_timestamps : sequence of (word, start, end)
        Word-level timestamps in the source audio.
    alignment : sequence of (source_word_idx, target_word_idx)
        Output of :func:`contextual_align.contextual_align`.
    min_lag : float
        Minimum delay (seconds) between source word onset and target word
        onset.
    sample_rate : int
        Sample rate of *target_audio*.

    Returns
    -------
    SilenceInsertionResult
    """
    target_audio = np.asarray(target_audio, dtype=np.float32).ravel()
    original_duration = _samples_to_seconds(len(target_audio), sample_rate)

    # Build alignment map: target_word_idx -> source_word_idx
    align_map: Dict[int, int] = {}
    for src_idx, tgt_idx in alignment:
        align_map[tgt_idx] = src_idx

    # Accumulate audio chunks (original segments + inserted silences)
    chunks: List[np.ndarray] = []
    inserted: List[Dict] = []
    modified_ts: List[WordTimestamp] = []

    cumulative_shift = 0.0  # total silence inserted so far (seconds)
    prev_end_sample = 0  # last sample copied from the original audio

    for j in range(len(target_timestamps)):
        word, tgt_start, tgt_end = target_timestamps[j]

        src_idx = align_map.get(j)
        if src_idx is not None and src_idx < len(source_timestamps):
            _src_word, src_start, _src_end = source_timestamps[src_idx]
            required_onset = src_start + min_lag
        else:
            # No alignment info -- keep original timing
            required_onset = None

        current_onset = tgt_start + cumulative_shift

        silence_dur = 0.0
        if required_onset is not None and current_onset < required_onset:
            silence_dur = required_onset - current_onset

        # Copy audio from prev_end_sample up to the start of this word
        word_start_sample = _seconds_to_samples(tgt_start, sample_rate)
        word_start_sample = max(prev_end_sample, min(word_start_sample, len(target_audio)))

        if word_start_sample > prev_end_sample:
            chunks.append(target_audio[prev_end_sample:word_start_sample])

        # Insert silence if needed
        if silence_dur > 0:
            chunks.append(_make_silence(silence_dur, sample_rate))
            inserted.append(
                {
                    "target_word_idx": j,
                    "word": word,
                    "insertion_point_sec": tgt_start + cumulative_shift,
                    "silence_duration_sec": silence_dur,
                }
            )
            cumulative_shift += silence_dur

        # Record shifted timestamp
        new_start = tgt_start + cumulative_shift
        new_end = tgt_end + cumulative_shift
        modified_ts.append((word, new_start, new_end))

        # Copy the word's audio segment
        word_end_sample = _seconds_to_samples(tgt_end, sample_rate)
        word_end_sample = max(word_start_sample, min(word_end_sample, len(target_audio)))
        if word_end_sample > word_start_sample:
            chunks.append(target_audio[word_start_sample:word_end_sample])
        prev_end_sample = word_end_sample

    # Append any remaining audio after the last word
    if prev_end_sample < len(target_audio):
        chunks.append(target_audio[prev_end_sample:])

    if chunks:
        modified_audio = np.concatenate(chunks)
    else:
        modified_audio = target_audio.copy()

    return SilenceInsertionResult(
        audio=modified_audio,
        original_duration=original_duration,
        modified_duration=_samples_to_seconds(len(modified_audio), sample_rate),
        inserted_silences=inserted,
        modified_timestamps=modified_ts,
    )


# ---------------------------------------------------------------------------
# Utterance-level convenience
# ---------------------------------------------------------------------------


def _load_audio(path: Union[str, Path], target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample to *target_sr* if necessary."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required -- pip install soundfile")

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(target_sr, sr)
            audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        except ImportError:
            raise ImportError("scipy is required for resampling -- pip install scipy")

    return audio


def process_utterance(
    source_audio_path: Union[str, Path],
    target_audio_path: Union[str, Path],
    source_text: str,
    target_text: str,
    alignment: Sequence[Tuple[int, int]],
    source_timestamps: Optional[Sequence[WordTimestamp]] = None,
    target_timestamps: Optional[Sequence[WordTimestamp]] = None,
    min_lag: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
) -> Dict:
    """High-level helper: load audio files, insert silences, return results.

    Parameters
    ----------
    source_audio_path, target_audio_path : path-like
        Paths to source and target audio files (wav, flac, etc.).
    source_text, target_text : str
        Whitespace-tokenised transcripts.
    alignment : sequence of (source_word_idx, target_word_idx)
        Contextual alignment pairs.
    source_timestamps, target_timestamps : optional
        Pre-computed word timestamps.  If *None* the caller must provide them
        separately (e.g. via a forced-alignment tool).
    min_lag : float
        Minimum lag in seconds.
    sample_rate : int
        Target sample rate.

    Returns
    -------
    dict with keys:
        - ``modified_audio``: np.ndarray -- silence-inserted waveform
        - ``original_audio``: np.ndarray -- original target waveform
        - ``source_audio``: np.ndarray -- source waveform (for reference)
        - ``result``: SilenceInsertionResult metadata
        - ``source_text``, ``target_text``, ``alignment``
    """
    source_audio = _load_audio(source_audio_path, target_sr=sample_rate)
    target_audio = _load_audio(target_audio_path, target_sr=sample_rate)

    if source_timestamps is None or target_timestamps is None:
        raise ValueError(
            "source_timestamps and target_timestamps are required. "
            "Use a forced-alignment tool (e.g. faster-whisper) to obtain "
            "word-level timestamps before calling process_utterance."
        )

    result = insert_silences(
        target_audio=target_audio,
        target_timestamps=target_timestamps,
        source_timestamps=source_timestamps,
        alignment=alignment,
        min_lag=min_lag,
        sample_rate=sample_rate,
    )

    return {
        "modified_audio": result.audio,
        "original_audio": target_audio,
        "source_audio": source_audio,
        "result": result,
        "source_text": source_text,
        "target_text": target_text,
        "alignment": list(alignment),
    }
