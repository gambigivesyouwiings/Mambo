"""Contextual alignment algorithm from Hibiki paper Section 3.2.1, eq. 6.

Computes per-word alignments between source and target sentences using the
perplexity of an NMT model. Given source sentence S = (S1, ..., Sn) and target
sentence T = (T1, ..., Tm), we find for each target word T_j the source word
index that maximally increases the conditional log-probability of T_j, then
apply spike smoothing and a minimum lag constraint.

The Hibiki paper uses MADLAD-400-3B for the perplexity model. We use
NLLB-200-distilled-1.3B instead because MADLAD's encoder/decoder produces
broken logits in our Kaggle T4 environment (see translate_nllb.py for the
diagnosis trail). The algorithm is model-agnostic; any encoder-decoder MT
model with reliable log-probabilities works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/nllb-200-distilled-1.3B"

# Short language code -> NLLB BCP-47-style code (vocabulary token).
NLLB_CODES = {
    "en": "eng_Latn",
    "sw": "swh_Latn",
    "swh": "swh_Latn",
    "eng_Latn": "eng_Latn",
    "swh_Latn": "swh_Latn",
}


def _resolve_lang(code: str) -> str:
    if code not in NLLB_CODES:
        raise ValueError(
            f"Unknown language {code!r}. Known: {sorted(set(NLLB_CODES))}. "
            "Add a mapping to NLLB_CODES if you need a new language."
        )
    return NLLB_CODES[code]


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_loaded_name = None


def _load_model(
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """Load (and cache) the NLLB translation model."""
    global _model, _tokenizer, _loaded_name
    if _model is not None and _loaded_name == model_name:
        return _model, _tokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading %s on %s", model_name, device)
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    _model.eval()
    _loaded_name = model_name
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _tokenize_words(text: str, tokenizer: AutoTokenizer) -> List[List[int]]:
    """Split *text* into whitespace words and return sub-token ids for each.

    Caller is responsible for setting tokenizer.src_lang for the encoder side
    (it does not affect per-word encoding here, which uses add_special_tokens=False).
    """
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
    src_lang_id: int,
    eos_id: int,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> _PrefixCache:
    """Encode source prefixes S1..Si for i in 1..n in a single batched pass.

    NLLB encoder input format is: [src_lang_id, ...source_tokens..., eos_id].
    """
    n = len(source_word_ids)
    prefix_seqs: List[List[int]] = []
    running: List[int] = []
    for i in range(n):
        running = running + source_word_ids[i]
        prefix_seqs.append([src_lang_id] + list(running) + [eos_id])

    max_len = max(len(s) for s in prefix_seqs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long, device=device)
    for idx, seq in enumerate(prefix_seqs):
        input_ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[idx, : len(seq)] = 1

    with torch.no_grad():
        enc = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)

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
    tgt_lang_id: int,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> np.ndarray:
    """Compute log P(T_j | S1..Si, T1..T_{j-1}) for all (i, j).

    NLLB decoder input format is: [decoder_start, tgt_lang_id, ...target_tokens...].
    Logits at position k predict the token at position k of the decoder output;
    position 0 predicts tgt_lang_id (skipped) and position k+1 predicts
    flat_target[k] for content tokens.
    """
    n = len(prefix_cache.encoder_outputs)
    m = len(target_word_ids)

    flat_target: List[int] = []
    word_starts: List[int] = []
    for toks in target_word_ids:
        word_starts.append(len(flat_target))
        flat_target.extend(toks)
    word_starts.append(len(flat_target))  # sentinel

    dec_start = model.config.decoder_start_token_id
    decoder_input = [dec_start, tgt_lang_id] + flat_target[:-1]
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
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        token_log_probs = torch.log_softmax(logits, dim=-1)

        for j in range(m):
            start = word_starts[j]
            end = word_starts[j + 1]
            word_lp = 0.0
            for t in range(start, end):
                # Decoder position predicting flat_target[t] is t+1
                # (offset by the prepended tgt_lang_id).
                word_lp += token_log_probs[t + 1, flat_target[t]].item()
            log_probs[i, j] = word_lp

    return log_probs


# ---------------------------------------------------------------------------
# Alignment extraction (eq. 6) + smoothing
# ---------------------------------------------------------------------------


def _extract_raw_alignment(log_probs: np.ndarray) -> np.ndarray:
    """For each target word j, find argmax_i [log p_{j,i} - log p_{j,i-1}]."""
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
    source_lang: str,
    target_lang: str,
    model_name: str = DEFAULT_MODEL,
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
    source_lang, target_lang : str
        Language codes ("sw"/"en" or full NLLB codes like "swh_Latn").
        Required because NLLB needs the source-lang token on the encoder side
        and the target-lang token on the decoder side.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    src_code = _resolve_lang(source_lang)
    tgt_code = _resolve_lang(target_lang)

    model, tokenizer = _load_model(model_name, device)
    tokenizer.src_lang = src_code
    src_lang_id = tokenizer.convert_tokens_to_ids(src_code)
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_code)
    eos_id = tokenizer.eos_token_id

    source_word_ids = _tokenize_words(source_text, tokenizer)
    target_word_ids = _tokenize_words(target_text, tokenizer)

    if not source_word_ids or not target_word_ids:
        return []

    prefix_cache = _encode_source_prefixes(
        source_word_ids, src_lang_id, eos_id, model, tokenizer, device
    )
    log_probs = _compute_log_probs(
        prefix_cache, target_word_ids, tgt_lang_id, model, tokenizer, device
    )

    alignment = _extract_raw_alignment(log_probs)
    alignment = _spike_smooth(alignment, window=spike_window, spike_threshold=spike_threshold)
    alignment = _enforce_min_lag(
        alignment, source_timestamps, target_timestamps, min_lag
    )

    return [(int(alignment[j]), j) for j in range(len(alignment))]


def batch_contextual_align(
    pairs: List[Tuple[str, str]],
    source_lang: str,
    target_lang: str,
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    spike_window: int = 5,
    spike_threshold: float = 0.25,
    min_lag: float = 2.0,
) -> List[List[Tuple[int, int]]]:
    """Process multiple (source, target) pairs efficiently.

    The model and tokenizer are loaded once and reused across all pairs.
    All pairs must share the same language direction.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    src_code = _resolve_lang(source_lang)
    tgt_code = _resolve_lang(target_lang)

    model, tokenizer = _load_model(model_name, device)
    tokenizer.src_lang = src_code
    src_lang_id = tokenizer.convert_tokens_to_ids(src_code)
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_code)
    eos_id = tokenizer.eos_token_id

    results: List[List[Tuple[int, int]]] = []
    for idx, (src, tgt) in enumerate(pairs):
        logger.info("Aligning pair %d / %d", idx + 1, len(pairs))
        src_word_ids = _tokenize_words(src, tokenizer)
        tgt_word_ids = _tokenize_words(tgt, tokenizer)

        if not src_word_ids or not tgt_word_ids:
            results.append([])
            continue

        prefix_cache = _encode_source_prefixes(
            src_word_ids, src_lang_id, eos_id, model, tokenizer, device
        )
        log_probs = _compute_log_probs(
            prefix_cache, tgt_word_ids, tgt_lang_id, model, tokenizer, device
        )

        alignment = _extract_raw_alignment(log_probs)
        alignment = _spike_smooth(alignment, window=spike_window, spike_threshold=spike_threshold)

        results.append([(int(alignment[j]), j) for j in range(len(alignment))])

    return results
