"""Dataset for Whisper ASR fine-tuning on Sw audio.

Combines:
  - KenSpeech (always loaded; assumed real labels)
  - Optional pseudo-label JSONL from filter_pseudo.py (or the unfiltered raw output)

Each item:
    {
      "input_features": (n_mels, T_in)   -- Whisper mel spectrogram
      "labels":         (L,)              -- Sw transcript token ids ending with EOT
                                            (HF Whisper trainer shifts these internally)
    }

The HuggingFace Whisper trainer convention is to pass ONLY `labels` and let the model
generate `decoder_input_ids` via shift_tokens_right(). We follow that convention here —
no manual decoder_input_ids construction (which is what bit us in whisper_st).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class SwASRDataset(Dataset):
    """KenSpeech (with real Sw transcripts) optionally augmented with pseudo-labeled audio.

    Args:
        kenspeech_dir: KenSpeech root.
        processor:     WhisperProcessor (language='sw', task='transcribe').
        pseudo_labels_path: Optional JSONL of {audio_path, pseudo_label, ...} from
            pseudo_label.py / filter_pseudo.py. If None, only KenSpeech is used.
        max_audio_seconds: Drop audios longer than this.
        max_label_tokens:  Drop samples with target longer than this token count.
    """

    def __init__(
        self,
        kenspeech_dir: str,
        processor,
        pseudo_labels_path: Optional[str] = None,
        max_audio_seconds: float = 28.0,
        max_label_tokens: int = 448,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_audio_samples = int(max_audio_seconds * 16000)
        self.max_label_tokens = max_label_tokens

        # ---- KenSpeech (real labels) ----
        sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "prepare"))
        from kenspeech_loader import KenSpeechLoader
        print(f"Loading KenSpeech index from {kenspeech_dir}...")
        self.kenspeech = KenSpeechLoader(load_audio=True, local_dir=kenspeech_dir)
        print(f"  KenSpeech: {len(self.kenspeech)} samples")

        # Build a flat list of (kind, payload) tuples so __getitem__ can dispatch
        self.entries: List[Dict] = []
        for sample_idx in range(len(self.kenspeech)):
            self.entries.append({"kind": "kenspeech", "sample_idx": sample_idx})

        # ---- Optional pseudo-labeled audio ----
        if pseudo_labels_path:
            n_added = 0
            with open(pseudo_labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    text = (e.get("pseudo_label") or "").strip()
                    if not text or not e.get("audio_path"):
                        continue
                    self.entries.append({
                        "kind": "pseudo",
                        "audio_path": e["audio_path"],
                        "text": text,
                    })
                    n_added += 1
            print(f"  Pseudo-labels: +{n_added} samples from {pseudo_labels_path}")

        print(f"  Total: {len(self.entries)} training samples")

    def __len__(self) -> int:
        return len(self.entries)

    def _load_audio(self, entry: Dict) -> tuple[np.ndarray, str]:
        if entry["kind"] == "kenspeech":
            sample = self.kenspeech[entry["sample_idx"]]
            audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
            text = (sample.get("transcription") or sample.get("text") or "").strip()
        else:
            audio, sr = sf.read(entry["audio_path"])
            if sr != 16000:
                # KenSpeech is already 16k; pseudo-audio we wrote at 16k. Defensive only.
                import torchaudio
                t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                t = torchaudio.functional.resample(t, sr, 16000)
                audio = t.squeeze(0).numpy()
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32, copy=False)
            text = entry["text"]
        if len(audio) > self.max_audio_samples:
            audio = audio[: self.max_audio_samples]
        return audio, text

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        audio, text = self._load_audio(entry)

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]  # (n_mels, T_in)

        # Tokenize with Whisper's special tokens for sw+transcribe.
        # processor.tokenizer(...) when configured with language='sw', task='transcribe'
        # adds [<|startoftranscript|>, <|sw|>, <|transcribe|>, <|notimestamps|>] then text + <|endoftext|>.
        # The trainer will shift these right internally to produce decoder_input_ids.
        labels = self.tokenizer(text, return_tensors="pt").input_ids[0]
        if labels.size(0) > self.max_label_tokens:
            labels = labels[: self.max_label_tokens]

        return {
            "input_features": input_features,
            "labels": labels,
        }


@dataclass
class WhisperASRCollator:
    """Pad labels to max length in batch with -100 (ignore_index for CE).

    input_features are fixed-shape (Whisper feature extractor pads/truncates to T_in=3000)
    so we can stack directly.
    """

    pad_token_id: int = 50257  # Whisper's pad

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([b["input_features"] for b in batch], dim=0)

        max_len = max(b["labels"].size(0) for b in batch)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["labels"].size(0)
            labels[i, :L] = b["labels"]

        return {"input_features": input_features, "labels": labels}
