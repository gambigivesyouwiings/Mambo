"""Mimi Neural Audio Codec wrapper (frozen, pretrained).

Wraps the kyutai/mimi codec from HuggingFace Transformers.
Encodes 24kHz mono audio -> discrete tokens at 12.5Hz with Q codebooks.
Decodes discrete tokens -> 24kHz mono audio.
"""

import torch
import torch.nn as nn
from typing import Optional


class MimiCodec(nn.Module):
    """Frozen wrapper around the pretrained Mimi neural audio codec.

    The codec encodes audio waveforms into discrete token sequences
    using Residual Vector Quantization (RVQ). Codebook 1 captures
    semantic information, codebooks 2+ capture acoustic details.
    """

    def __init__(
        self,
        model_name: str = "kyutai/mimi",
        num_codebooks: int = 8,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_codebooks = num_codebooks
        self.sample_rate = 24000
        self.frame_rate = 12.5  # Hz
        self._device = device
        self._model = None  # lazy load

    def _load_model(self):
        """Lazy-load the Mimi model to avoid loading at import time."""
        if self._model is not None:
            return

        from transformers import MimiModel

        self._model = MimiModel.from_pretrained(self.model_name)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

        if self._device is not None:
            self._model = self._model.to(self._device)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to discrete tokens.

        Args:
            audio: (batch, 1, samples) or (batch, samples) tensor at 24kHz

        Returns:
            tokens: (batch, num_codebooks, time_steps) integer tensor
        """
        self._load_model()

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # add channel dim

        device = next(self._model.parameters()).device
        audio = audio.to(device)

        encoder_outputs = self._model.encode(audio, num_quantizers=self.num_codebooks)
        # audio_codes shape: (batch, num_codebooks, time_steps)
        tokens = encoder_outputs.audio_codes
        return tokens[:, : self.num_codebooks, :]

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens back to audio waveform.

        Args:
            tokens: (batch, num_codebooks, time_steps) integer tensor

        Returns:
            audio: (batch, 1, samples) tensor at 24kHz
        """
        self._load_model()

        device = next(self._model.parameters()).device
        tokens = tokens.to(device)

        # Pad to expected codebook count if needed
        if tokens.shape[1] < self._model.config.num_quantizers:
            pad = torch.zeros(
                tokens.shape[0],
                self._model.config.num_quantizers - tokens.shape[1],
                tokens.shape[2],
                dtype=tokens.dtype,
                device=device,
            )
            tokens = torch.cat([tokens, pad], dim=1)

        audio = self._model.decode(tokens)
        return audio.audio_values

    @property
    def codebook_size(self) -> int:
        """Number of entries per codebook."""
        return 2048

    def samples_to_frames(self, num_samples: int) -> int:
        """Convert number of audio samples to number of codec frames."""
        return int(num_samples * self.frame_rate / self.sample_rate)

    def frames_to_samples(self, num_frames: int) -> int:
        """Convert number of codec frames to number of audio samples."""
        return int(num_frames * self.sample_rate / self.frame_rate)
