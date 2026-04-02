"""Contextual alignment algorithm from Hibiki paper Section 3.2.1, eq. 6.

Computes per-word alignments between source and target sentences using the
perplexity of the MADLAD-400-3B machine translation model.  Given source
sentence S = (S1, ..., Sn) and target sentence T = (T1, ..., Tm), we find
for each target word T_j the source word index that maximally increases the
conditional log-probability of T_j, then apply spike smoothing and a minimum
lag constraint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None


def _load_model(
    model_name: str = "google/madlad400-3b-mt",
    device: Optional[str] = None,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """Load (and cache) the MADLAD-400 translation model."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading %s on %s", model_name, device)
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    _model.eval()
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _tokenize_words(text: str, tokenizer: AutoTokenizer) -> List[List[int]]:
    """Split *text* into whitespace words and return sub-token ids for each."""
    words = text.strip().split()
    word_token_ids: List[List[int]] = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        word_token_ids.append(ids)
    return word_token_ids


@dataclass
class _PrefixCache:
    """Holds encoder outputs for incrementally growing source prefixes."""
    encoder_outputs: list  # one per prefix length (1..n)


def _encode_source_prefixes(
    source_word_ids: List[List[int]],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> _PrefixCache:
    """Encode source prefixes S1..Si for i in 1..n in a single batched pass."""
    n = len(source_word_ids)
    # Build token sequences for each prefix length
    prefix_seqs: List[List[int]] = []
    running: List[int] = []
    for i in range(n):
        running = running + source_word_ids[i]
        prefix_seqs.append(list(running))

    # Pad to equal length for batched encoding
    max_len = max(len(s) for s in prefix_seqs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long, device=device)
    for idx, seq in enumerate(prefix_seqs):
        input_ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[idx, : len(seq)] = 1

    with torch.no_grad():
        enc = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)

    # Store per-prefix encoder outputs + masks
    cache = _PrefixCache(encoder_outputs=[])
    for idx in range(n):
        length = len(prefix_seqs[idx])
        cache.encoder_outputs.append(
            {
                "last_hidden_state": enc.last_hidden_state[idx : idx + 1, :length, :],
                "attention_mask": attention_mask[idx : idx + 1, :length],
            }
        )
    return cache


def _compute_log_probs(
    prefix_cache: _PrefixCache,
    target_word_ids: List[List[int]],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> np.ndarray:
    """Compute log P(T_j | S1..Si, T1..T_{j-1}) for all (i, j).

    Returns an array of shape (n_source_words, n_target_words) where entry
    [i, j] is the log-probability of the j-th target word conditioned on
    source prefix of length i+1 and target prefix T1..T_{j-1}.
    """
    n = len(prefix_cache.encoder_outputs)
    m = len(target_word_ids)

    # Flatten target tokens and record word boundaries
    flat_target: List[int] = []
    word_starts: List[int] = []  # index into flat_target where each word begins
    for toks in target_word_ids:
        word_starts.append(len(flat_target))
        flat_target.extend(toks)
    word_starts.append(len(flat_target))  # sentinel

    # Prepare decoder input: shift right (prepend decoder_start_token_id)
    dec_start = model.config.decoder_start_token_id
    decoder_input = [dec_start] + flat_target[:-1]
    decoder_ids = torch.tensor([decoder_input], dtype=torch.long, device=device)

    log_probs = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        enc_out = prefix_cache.encoder_outputs[i]
        encoder_hidden = enc_out["last_hidden_state"]
        encoder_mask = enc_out["attention_mask"]

        with torch.no_grad():
            outputs = model(
                encoder_outputs=(encoder_hidden,),
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_ids,
            )
        # outputs.logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        token_log_probs = torch.log_softmax(logits, dim=-1)

        # Aggregate per-word log-prob as sum of sub-token log-probs
        for j in range(m):
            start = word_starts[j]
            end = word_starts[j + 1]
            word_lp = 0.0
            for t in range(start, end):
                word_lp += token_log_probs[t, flat_target[t]].item()
            log_probs[i, j] = word_lp

    return log_probs


# ---------------------------------------------------------------------------
# Alignment extraction (eq. 6) + smoothing
# ---------------------------------------------------------------------------


def _extract_raw_alignment(log_probs: np.ndarray) -> np.ndarray:
    """For each target word j, find argmax_i [log p_{j,i} - log p_{j,i-1}].

    Parameters
    ----------
    log_probs : ndarray of shape (n, m)

    Returns
    -------
    alignment : ndarray of shape (m,) with source word indices (0-based).
    """
    n, m = log_probs.shape
    alignment = np.zeros(m, dtype=np.int64)
    for j in range(m):
        best_i = 0
        best_delta = -np.inf
        for i in range(n):
            prev = log_probs[i - 1, j] if i > 0 else 0.0
            delta = log_probs[i, j] - prev
            if delta > best_delta:
                best_delta = delta
                best_i = i
        alignment[j] = best_i
    return alignment


def _spike_smooth(
    alignment: np.ndarray,
    window: int = 5,
    spike_threshold: float = 0.25,
) -> np.ndarray:
    """Remove alignment spikes that exceed *spike_threshold* of the local
    average delay over a sliding window of *window* words.

    A spike is a point where the alignment jumps more than
    ``spike_threshold * local_avg`` compared to its neighbours; we replace
    it with linear interpolation from the surrounding values.
    """
    alignment = alignment.astype(np.float64).copy()
    m = len(alignment)
    if m < 3:
        return alignment.astype(np.int64)

    half = window // 2
    smoothed = alignment.copy()

    for j in range(1, m - 1):
        lo = max(0, j - half)
        hi = min(m, j + half + 1)
        local_avg = np.mean(alignment[lo:hi])
        if local_avg == 0:
            continue
        deviation = abs(alignment[j] - (alignment[j - 1] + alignment[j + 1]) / 2.0)
        if deviation > spike_threshold * local_avg:
            smoothed[j] = (alignment[j - 1] + alignment[j + 1]) / 2.0

    return np.clip(np.round(smoothed), 0, None).astype(np.int64)


def _enforce_min_lag(
    alignment: np.ndarray,
    source_timestamps: Optional[Sequence[float]] = None,
    target_timestamps: Optional[Sequence[float]] = None,
    min_lag: float = 2.0,
) -> np.ndarray:
    """Ensure every aligned pair respects the minimum lag constraint.

    If timestamps are not provided the alignment is returned as-is (the lag
    constraint is enforced downstream in ``silence_insertion``).
    """
    if source_timestamps is None or target_timestamps is None:
        return alignment

    alignment = alignment.copy()
    n_src = len(source_timestamps)
    for j in range(len(alignment)):
        src_i = int(alignment[j])
        if src_i >= n_src:
            src_i = n_src - 1
            alignment[j] = src_i
        src_time = source_timestamps[src_i]
        tgt_time = target_timestamps[j] if j < len(target_timestamps) else 0.0
        # If target word is too early, push alignment to an earlier source word
        while src_i > 0 and tgt_time < src_time + min_lag:
            src_i -= 1
            src_time = source_timestamps[src_i]
        alignment[j] = src_i
    return alignment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def contextual_align(
    source_text: str,
    target_text: str,
    model_name: str = "google/madlad400-3b-mt",
    device: Optional[str] = None,
    spike_window: int = 5,
    spike_threshold: float = 0.25,
    source_timestamps: Optional[Sequence[float]] = None,
    target_timestamps: Optional[Sequence[float]] = None,
    min_lag: float = 2.0,
) -> List[Tuple[int, int]]:
    """Compute contextual alignment between *source_text* and *target_text*.

    Parameters
    ----------
    source_text : str
        Source-language sentence (whitespace-tokenised).
    target_text : str
        Target-language sentence (whitespace-tokenised).
    model_name : str
        HuggingFace model identifier for the translation model.
    device : str or None
        Torch device string; auto-detected if *None*.
    spike_window, spike_threshold : int, float
        Parameters for spike smoothing.
    source_timestamps, target_timestamps : sequence of float or None
        Optional per-word onset times (seconds) used for the minimum lag
        constraint.  If not provided the lag check is deferred.
    min_lag : float
        Minimum allowed lag in seconds between aligned source word and
        target word.

    Returns
    -------
    alignment : list of (source_word_idx, target_word_idx)
        One pair per target word.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = _load_model(model_name, device)

    source_word_ids = _tokenize_words(source_text, tokenizer)
    target_word_ids = _tokenize_words(target_text, tokenizer)

    if not source_word_ids or not target_word_ids:
        return []

    prefix_cache = _encode_source_prefixes(source_word_ids, model, tokenizer, device)
    log_probs = _compute_log_probs(prefix_cache, target_word_ids, model, tokenizer, device)

    alignment = _extract_raw_alignment(log_probs)
    alignment = _spike_smooth(alignment, window=spike_window, spike_threshold=spike_threshold)
    alignment = _enforce_min_lag(
        alignment, source_timestamps, target_timestamps, min_lag
    )

    return [(int(alignment[j]), j) for j in range(len(alignment))]


def batch_contextual_align(
    pairs: List[Tuple[str, str]],
    model_name: str = "google/madlad400-3b-mt",
    device: Optional[str] = None,
    spike_window: int = 5,
    spike_threshold: float = 0.25,
    min_lag: float = 2.0,
) -> List[List[Tuple[int, int]]]:
    """Process multiple (source, target) pairs efficiently.

    The translation model is loaded once and reused across all pairs.  Source
    prefix encoding is batched per sentence.

    Parameters
    ----------
    pairs : list of (source_text, target_text)
    model_name, device, spike_window, spike_threshold, min_lag :
        Same semantics as :func:`contextual_align`.

    Returns
    -------
    alignments : list of alignment lists, one per input pair.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure model is loaded once
    model, tokenizer = _load_model(model_name, device)

    results: List[List[Tuple[int, int]]] = []
    for idx, (src, tgt) in enumerate(pairs):
        logger.info("Aligning pair %d / %d", idx + 1, len(pairs))
        src_word_ids = _tokenize_words(src, tokenizer)
        tgt_word_ids = _tokenize_words(tgt, tokenizer)

        if not src_word_ids or not tgt_word_ids:
            results.append([])
            continue

        prefix_cache = _encode_source_prefixes(src_word_ids, model, tokenizer, device)
        log_probs = _compute_log_probs(prefix_cache, tgt_word_ids, model, tokenizer, device)

        alignment = _extract_raw_alignment(log_probs)
        alignment = _spike_smooth(alignment, window=spike_window, spike_threshold=spike_threshold)

        results.append([(int(alignment[j]), j) for j in range(len(alignment))])

    return results
