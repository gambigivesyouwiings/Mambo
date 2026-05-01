"""Dataset for transcript-prompted Whisper training on KenSpeech x NLLB pairs.

Reads translation JSONs produced by the data pipeline (translate_nllb.py) and
joins them with KenSpeech audio to produce training tuples:

    {
        "input_features": (n_mels, T_in)    -- Whisper mel spectrogram
        "decoder_input_ids": (L,)           -- full prompt incl. transcript prefix
        "labels": (L,)                       -- CE supervision (transcript masked -100)
        "transcript_labels": (T_max,)        -- CTC supervision (Sw transcript only)
        "transcript_label_lengths": int     -- actual length of transcript_labels
    }

Decoder prompt format (matches Whisper's special-token convention):
    <|sot|><|sw|><|transcribe|>{transcript}{optional lexicon hint}<|en|><|translate|>{translation}<|eot|>

When a lexicon is supplied, with probability `hint_prob` we look up content
words in the source transcript against the lexicon and append a parenthetical
hint (e.g. "transcript (nairobi=nairobi, mji mkuu=capital city)"). The model
learns to attend to these hints during training so they are useful at inference.
The CTC supervision target stays clean (transcript only, no hint) since CTC
operates on the encoder output and supervises the actual spoken transcript.

Loss masking: positions in `labels` corresponding to the transcript prefix and
the special tokens are set to -100 so the decoder is supervised only on the
translation portion. The CTC loss handles the transcript supervision.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def load_lexicon_dict(path: Optional[str]) -> Optional[Dict[str, str]]:
    """Load a JSONL lexicon into {sw: en} dict, lowercased keys. None if path is None."""
    if not path:
        return None
    lex: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                lex[e["sw"].lower()] = e["en"]
            except Exception:
                continue
    return lex


def lookup_lexicon_hits(transcript: str, lexicon: Dict[str, str], max_hits: int = 6) -> List[Dict[str, str]]:
    """Find Sw words/bigrams in transcript that have lexicon entries.

    Tries bigrams first (more specific), then unigrams. Returns up to max_hits
    unique matches in transcript order. Same logic used at inference time.
    """
    hits = []
    seen = set()
    words = WORD_RE.findall(transcript.lower())
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        if bg in lexicon and bg not in seen:
            hits.append({"sw": bg, "en": lexicon[bg]})
            seen.add(bg)
    for w in words:
        if w in lexicon and w not in seen:
            hits.append({"sw": w, "en": lexicon[w]})
            seen.add(w)
    return hits[:max_hits]


def format_lexicon_hint(hits: List[Dict[str, str]]) -> str:
    """Format lexicon matches as parenthetical hint, or empty string if no hits."""
    if not hits:
        return ""
    pairs = ", ".join(f"{h['sw']}={h['en']}" for h in hits)
    return f" ({pairs})"


class KenSpeechSTDataset(Dataset):
    """Speech translation dataset (Sw audio -> Sw transcript -> En translation).

    Args:
        translations_dir: Directory of *_en.json files from translate_nllb.py.
        kenspeech_dir: Local KenSpeech root (with metadata.csv + audio/).
        processor: A WhisperProcessor for feature extraction + tokenization.
        max_audio_seconds: Drop samples longer than this (Whisper caps at 30s).
        max_label_tokens: Drop samples whose decoder prompt exceeds this length.
        lexicon_path: Optional path to JSONL lexicon. If set, training-time
            hint injection is enabled.
        hint_prob: Probability per sample of injecting lexicon hints when a
            lexicon is loaded (0.0 = never, 1.0 = always). Default 0.5 trains
            the model to handle both with-hint and no-hint prompts.
    """

    def __init__(
        self,
        translations_dir: str,
        kenspeech_dir: str,
        processor,
        max_audio_seconds: float = 28.0,
        max_label_tokens: int = 384,
        lexicon_path: Optional[str] = None,
        hint_prob: float = 0.5,
    ):
        # Lazy import to avoid hard dependency unless dataset is constructed
        sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "prepare"))
        from kenspeech_loader import KenSpeechLoader

        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_audio_samples = int(max_audio_seconds * 16000)
        self.max_label_tokens = max_label_tokens

        # Build sample_idx -> KenSpeech sample mapping (load_audio=True so we
        # can fetch waveforms by index).
        print(f"Loading KenSpeech index from {kenspeech_dir}...")
        self.kenspeech = KenSpeechLoader(load_audio=True, local_dir=kenspeech_dir)
        print(f"KenSpeech: {len(self.kenspeech)} samples")

        # Optional: load lexicon for training-time hint injection
        self.lexicon = load_lexicon_dict(lexicon_path)
        self.hint_prob = hint_prob if self.lexicon else 0.0
        if self.lexicon:
            print(f"Loaded lexicon ({len(self.lexicon)} entries) for training-time "
                  f"hint injection (p={self.hint_prob})")

        # Index translation JSONs by sample_idx for fast joining
        print(f"Indexing translations from {translations_dir}...")
        self.examples: List[Dict] = []
        json_files = sorted(Path(translations_dir).glob("*.json"))
        for jp in json_files:
            if jp.name == "index.jsonl":
                continue
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            sample_idx = data.get("sample_idx", -1)
            sw_text = (data.get("source_text") or "").strip()
            en_text = (data.get("translated_text") or "").strip()
            if sample_idx < 0 or not sw_text or not en_text:
                continue
            self.examples.append({
                "sample_idx": sample_idx,
                "sw_text": sw_text,
                "en_text": en_text,
            })
        print(f"Built {len(self.examples)} (audio, sw, en) training examples")

        # Pre-compute special token ids used in the decoder prompt
        # Whisper's tokenizer exposes language and task token IDs via lookup.
        try:
            self.sot_id = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
            self.sw_id = self.tokenizer.convert_tokens_to_ids("<|sw|>")
            self.en_id = self.tokenizer.convert_tokens_to_ids("<|en|>")
            self.transcribe_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
            self.translate_id = self.tokenizer.convert_tokens_to_ids("<|translate|>")
            self.notimestamps_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            self.eot_id = self.tokenizer.eos_token_id
        except Exception as e:
            raise RuntimeError(
                f"Could not resolve Whisper special tokens via the tokenizer: {e}. "
                "Ensure the processor is a WhisperProcessor."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]

        # Fetch audio
        sample = self.kenspeech[ex["sample_idx"]]
        audio = sample["audio"]["array"]
        if len(audio) > self.max_audio_samples:
            # Truncate (rare; Whisper's feature extractor caps at 30s anyway)
            audio = audio[: self.max_audio_samples]
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32, copy=False)

        # Mel spectrogram
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]  # (n_mels, T_in)

        # Tokenize translation (no special tokens; we add them)
        translation_ids = self.tokenizer(ex["en_text"], add_special_tokens=False).input_ids

        # Clean transcript ids — used for the CTC supervision target
        ctc_transcript_ids = self.tokenizer(ex["sw_text"], add_special_tokens=False).input_ids

        # Decoder-side transcript may include a lexicon hint (training-time augmentation)
        sw_text_for_decoder = ex["sw_text"]
        if self.lexicon and self.hint_prob > 0 and random.random() < self.hint_prob:
            hits = lookup_lexicon_hits(ex["sw_text"], self.lexicon)
            sw_text_for_decoder = ex["sw_text"] + format_lexicon_hint(hits)
        transcript_ids = self.tokenizer(sw_text_for_decoder, add_special_tokens=False).input_ids

        # Build decoder input: [SOT, sw, transcribe, notimestamps, ...transcript[+hint]..., en, translate, notimestamps, ...translation..., EOT]
        prefix = [self.sot_id, self.sw_id, self.transcribe_id, self.notimestamps_id]
        switch = [self.en_id, self.translate_id, self.notimestamps_id]

        decoder_input_ids = (
            prefix + transcript_ids + switch + translation_ids + [self.eot_id]
        )

        # Labels: at position i we want the model to predict decoder_input_ids[i+1]
        # (next-token objective). Mask everything except positions whose target is
        # in the translation portion or the final EOT. Last position has no next
        # token so it's masked.
        prefix_len = len(prefix) + len(transcript_ids) + len(switch)
        labels = [-100] * len(decoder_input_ids)
        # Position prefix_len-1 (input = last switch token "notimestamps") predicts translation_ids[0]
        # Position prefix_len+k (input = translation_ids[k]) predicts translation_ids[k+1]
        # Position len-2 (input = last translation token) predicts EOT
        # Position len-1 (input = EOT) has no successor — stays -100
        for i in range(prefix_len - 1, len(decoder_input_ids) - 1):
            labels[i] = decoder_input_ids[i + 1]

        # Truncate if too long
        if len(decoder_input_ids) > self.max_label_tokens:
            decoder_input_ids = decoder_input_ids[: self.max_label_tokens]
            labels = labels[: self.max_label_tokens]

        # CTC labels are the CLEAN transcript (no hint), since CTC supervises
        # the encoder's transcription of the actual spoken audio.
        transcript_labels = ctc_transcript_ids
        transcript_label_length = len(transcript_labels)

        return {
            "input_features": input_features,
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "transcript_labels": torch.tensor(transcript_labels, dtype=torch.long),
            "transcript_label_length": torch.tensor(transcript_label_length, dtype=torch.long),
        }


class TranscriptPromptedCollator:
    """Pad variable-length tensors in a batch.

    - input_features: stack as-is (Whisper feature extractor produces fixed T_in=3000)
    - decoder_input_ids / labels: pad to max length in batch
    - transcript_labels: pad to max length in batch (with 0 = blank for CTC)
    """

    def __init__(self, pad_token_id: int = 50257, ctc_blank_id: int = 0):
        self.pad_token_id = pad_token_id
        self.ctc_blank_id = ctc_blank_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # input_features
        input_features = torch.stack([b["input_features"] for b in batch], dim=0)

        # decoder_input_ids and labels (pad with pad_token / -100 respectively)
        max_len = max(b["decoder_input_ids"].size(0) for b in batch)
        decoder_input_ids = torch.full(
            (len(batch), max_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        decoder_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["decoder_input_ids"].size(0)
            decoder_input_ids[i, :L] = b["decoder_input_ids"]
            labels[i, :L] = b["labels"]
            decoder_attention_mask[i, :L] = 1

        # transcript_labels (pad with -100; will be masked to blank inside model)
        max_t = max(int(b["transcript_label_length"]) for b in batch)
        transcript_labels = torch.full((len(batch), max_t), -100, dtype=torch.long)
        transcript_label_lengths = torch.zeros(len(batch), dtype=torch.long)
        for i, b in enumerate(batch):
            L = int(b["transcript_label_length"])
            transcript_labels[i, :L] = b["transcript_labels"][:L]
            transcript_label_lengths[i] = L

        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "transcript_labels": transcript_labels,
            "transcript_label_lengths": transcript_label_lengths,
        }
