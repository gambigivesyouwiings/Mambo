"""Full Hibiki Model — combines Temporal and Depth Transformers for
end-to-end speech-to-speech translation.

The model processes source audio tokens and generates target audio tokens
plus an aligned text stream (Inner Monologue) for simultaneous translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .temporal_transformer import TemporalTransformer
from .depth_transformer import DepthTransformer


class HibikiModel(nn.Module):
    """Hibiki: End-to-end simultaneous speech translation model.

    Architecture:
        - Mimi codec (external, frozen): audio <-> discrete tokens
        - Temporal Transformer: processes token sequences over time
        - Depth Transformer: generates hierarchical codebook tokens per timestep
        - Inner Monologue: aligned text stream predicted jointly with audio

    Token delay pattern (per Hibiki eq. 3):
        - Semantic tokens (q=1): no delay
        - Acoustic tokens (q>=2): delayed by 2 timesteps
    """

    def __init__(
        self,
        d_model: int = 512,
        ffn_dim: int = 1408,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64,
        max_seq_len: int = 250,
        local_window: int = 250,
        depth_d_model: int = 384,
        depth_ffn_dim: int = 1024,
        depth_layers_per_cb: int = 4,
        num_codebooks: int = 8,
        weight_sharing_start: int = 5,
        text_vocab_size: int = 32000,
        audio_codebook_size: int = 2048,
        num_voice_categories: int = 5,
        acoustic_delay: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.acoustic_delay = acoustic_delay
        self.text_vocab_size = text_vocab_size
        self.audio_codebook_size = audio_codebook_size

        # Temporal Transformer
        self.temporal = TemporalTransformer(
            d_model=d_model,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            local_window=local_window,
            dropout=dropout,
            text_vocab_size=text_vocab_size,
            audio_codebook_size=audio_codebook_size,
            num_codebooks=num_codebooks,
        )

        # Depth Transformer
        self.depth = DepthTransformer(
            d_model=depth_d_model,
            ffn_dim=depth_ffn_dim,
            num_layers_per_codebook=depth_layers_per_cb,
            num_codebooks=num_codebooks,
            num_streams=2,
            weight_sharing_start=weight_sharing_start,
            audio_codebook_size=audio_codebook_size,
            temporal_d_model=d_model,
            dropout=dropout,
        )

        # Voice transfer conditioning embeddings
        self.voice_cond_embed = nn.Embedding(num_voice_categories, d_model)

        # Special token IDs
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.epad_id = 3

    def apply_acoustic_delay(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply acoustic delay pattern (Hibiki eq. 3).

        Semantic tokens (q=0) have no delay.
        Acoustic tokens (q>=1) are delayed by self.acoustic_delay steps.

        Args:
            tokens: (B, Q, T) audio tokens

        Returns:
            delayed: (B, Q, T) delayed tokens with zero-padding at start
        """
        B, Q, T = tokens.shape
        delayed = tokens.clone()
        if Q > 1 and self.acoustic_delay > 0:
            d = self.acoustic_delay
            # Shift acoustic tokens right by d steps, fill with pad
            delayed[:, 1:, d:] = tokens[:, 1:, :-d]
            delayed[:, 1:, :d] = self.pad_id
        return delayed

    def remove_acoustic_delay(self, tokens: torch.Tensor) -> torch.Tensor:
        """Remove acoustic delay before codec decoding.

        Args:
            tokens: (B, Q, T) delayed audio tokens

        Returns:
            undelayed: (B, Q, T) tokens with delay removed
        """
        B, Q, T = tokens.shape
        undelayed = tokens.clone()
        if Q > 1 and self.acoustic_delay > 0:
            d = self.acoustic_delay
            undelayed[:, 1:, :-d] = tokens[:, 1:, d:]
            undelayed[:, 1:, -d:] = self.pad_id
        return undelayed

    def forward(
        self,
        target_audio_tokens: torch.Tensor,
        source_audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        voice_category: Optional[torch.Tensor] = None,
        predict_source: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing.

        Args:
            target_audio_tokens: (B, Q, T) target audio tokens (already delayed)
            source_audio_tokens: (B, Q, T) source audio tokens (already delayed)
            text_tokens: (B, T) aligned text tokens (inner monologue)
            voice_category: (B,) voice similarity category index
            predict_source: whether to predict source stream tokens

        Returns:
            dict with keys:
                text_logits: (B, T, text_vocab+4)
                target_audio_logits: (B, Q, T, codebook_size+4)
                source_audio_logits: (B, Q, T, codebook_size+4) or None
        """
        B, Q, T = target_audio_tokens.shape

        # Shift tokens right by 1 for teacher forcing (input is t-1 tokens)
        # At t=0, use BOS tokens
        tgt_input = torch.full_like(target_audio_tokens, self.bos_id)
        tgt_input[:, :, 1:] = target_audio_tokens[:, :, :-1]

        src_input = torch.full_like(source_audio_tokens, self.bos_id)
        src_input[:, :, 1:] = source_audio_tokens[:, :, :-1]

        text_input = torch.full_like(text_tokens, self.bos_id)
        text_input[:, 1:] = text_tokens[:, :-1]

        # Voice conditioning
        voice_embed = None
        if voice_category is not None:
            vc = self.voice_cond_embed(voice_category)  # (B, d_model)
            voice_embed = vc.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_model)

        # Embed all input tokens
        h = self.temporal.embed_tokens(
            text_tokens=text_input,
            target_audio_tokens=tgt_input,
            source_audio_tokens=src_input,
            voice_cond_embed=voice_embed,
        )

        # Temporal Transformer
        z, text_logits, _ = self.temporal(h)

        # Depth Transformer
        text_embed = self.temporal.text_embed(text_input)
        target_audio_logits, source_audio_logits = self.depth(
            z,
            target_tokens=target_audio_tokens if predict_source else target_audio_tokens,
            source_tokens=source_audio_tokens if predict_source else None,
            text_embed=text_embed,
        )

        return {
            "text_logits": text_logits,
            "target_audio_logits": target_audio_logits,
            "source_audio_logits": source_audio_logits,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_audio_tokens: torch.Tensor,
        source_audio_tokens: Optional[torch.Tensor],
        text_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute cross-entropy losses for all streams.

        Args:
            outputs: dict from forward()
            target_audio_tokens: (B, Q, T) ground truth target tokens
            source_audio_tokens: (B, Q, T) ground truth source tokens
            text_tokens: (B, T) ground truth text tokens

        Returns:
            dict with loss components and total loss
        """
        # Text loss
        text_logits = outputs["text_logits"]  # (B, T, V)
        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.shape[-1]),
            text_tokens.reshape(-1),
            ignore_index=self.pad_id,
        )

        # Target audio loss (all codebooks)
        tgt_logits = outputs["target_audio_logits"]  # (B, Q, T, V)
        B, Q, T, V = tgt_logits.shape
        tgt_loss = F.cross_entropy(
            tgt_logits.reshape(-1, V),
            target_audio_tokens.reshape(-1),
            ignore_index=self.pad_id,
        )

        losses = {
            "text_loss": text_loss,
            "target_audio_loss": tgt_loss,
            "total_loss": text_loss + tgt_loss,
        }

        # Source audio loss (training only)
        if outputs["source_audio_logits"] is not None and source_audio_tokens is not None:
            src_logits = outputs["source_audio_logits"]
            src_loss = F.cross_entropy(
                src_logits.reshape(-1, src_logits.shape[-1]),
                source_audio_tokens.reshape(-1),
                ignore_index=self.pad_id,
            )
            losses["source_audio_loss"] = src_loss
            losses["total_loss"] = losses["total_loss"] + src_loss

        return losses

    @torch.no_grad()
    def generate(
        self,
        source_audio_tokens: torch.Tensor,
        max_len: int = 250,
        temperature: float = 0.8,
        top_k_audio: int = 250,
        top_k_text: int = 50,
        text_temperature: float = 0.8,
        voice_category: Optional[torch.Tensor] = None,
        cfg_gamma: float = 1.0,
        cfg_bad_category: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive generation for inference.

        Processes source audio tokens one frame at a time and generates
        target audio + text tokens.

        Args:
            source_audio_tokens: (B, Q, T_src) source audio tokens
            max_len: maximum output length in frames
            temperature: audio sampling temperature
            top_k_audio: top-k for audio tokens
            top_k_text: top-k for text tokens
            text_temperature: text sampling temperature
            voice_category: (B,) voice similarity category
            cfg_gamma: classifier-free guidance weight (1.0 = no CFG)
            cfg_bad_category: (B,) "bad" category for CFG

        Returns:
            generated_audio: (B, Q, T_out) generated target audio tokens
            generated_text: (B, T_out) generated text tokens
        """
        B, Q, T_src = source_audio_tokens.shape
        device = source_audio_tokens.device

        # Apply delay to source tokens
        src_delayed = self.apply_acoustic_delay(source_audio_tokens)

        # Voice conditioning
        voice_embed = None
        if voice_category is not None:
            voice_embed = self.voice_cond_embed(voice_category)  # (B, d_model)

        # Initialize outputs
        all_audio_tokens = []
        all_text_tokens = []
        kv_caches = None

        # BOS tokens for first step
        prev_tgt_audio = torch.full((B, Q), self.bos_id, device=device, dtype=torch.long)
        prev_text = torch.full((B,), self.bos_id, device=device, dtype=torch.long)

        for t in range(min(max_len, T_src)):
            # Get source tokens at time t
            src_t = src_delayed[:, :, t]  # (B, Q)

            # Embed tokens
            voice_t = voice_embed.unsqueeze(1) if voice_embed is not None else None
            h = self.temporal.embed_tokens(
                text_tokens=prev_text.unsqueeze(1),
                target_audio_tokens=prev_tgt_audio.unsqueeze(2),
                source_audio_tokens=src_t.unsqueeze(2),
                voice_cond_embed=voice_t,
            )

            # Temporal transformer step
            z, text_logits, kv_caches = self.temporal(h, offset=t, kv_caches=kv_caches)

            # Sample text token
            text_logits_t = text_logits[:, 0, :] / text_temperature
            if top_k_text > 0:
                topk_vals, _ = text_logits_t.topk(top_k_text, dim=-1)
                text_logits_t[text_logits_t < topk_vals[:, -1:]] = float("-inf")
            text_probs = F.softmax(text_logits_t, dim=-1)
            text_tok = torch.multinomial(text_probs, 1).squeeze(-1)
            all_text_tokens.append(text_tok)

            # Depth transformer: generate audio tokens
            text_embed_t = self.temporal.text_embed(text_tok.unsqueeze(1))
            audio_tok = self.depth.step(
                z, text_embed_t=text_embed_t,
                temperature=temperature, top_k=top_k_audio,
            )
            all_audio_tokens.append(audio_tok)

            # Update previous tokens for next step
            prev_tgt_audio = audio_tok
            prev_text = text_tok

            # Check for EOS in text stream
            if (text_tok == self.eos_id).all():
                break

        # Continue generating after source ends (model produces its own EOS)
        # Send source EOS tokens
        src_eos = torch.full((B, Q), self.eos_id, device=device, dtype=torch.long)
        for t_extra in range(T_src, max_len):
            voice_t = voice_embed.unsqueeze(1) if voice_embed is not None else None
            h = self.temporal.embed_tokens(
                text_tokens=prev_text.unsqueeze(1),
                target_audio_tokens=prev_tgt_audio.unsqueeze(2),
                source_audio_tokens=src_eos.unsqueeze(2),
                voice_cond_embed=voice_t,
            )

            z, text_logits, kv_caches = self.temporal(h, offset=t_extra, kv_caches=kv_caches)

            text_logits_t = text_logits[:, 0, :] / text_temperature
            if top_k_text > 0:
                topk_vals, _ = text_logits_t.topk(top_k_text, dim=-1)
                text_logits_t[text_logits_t < topk_vals[:, -1:]] = float("-inf")
            text_probs = F.softmax(text_logits_t, dim=-1)
            text_tok = torch.multinomial(text_probs, 1).squeeze(-1)
            all_text_tokens.append(text_tok)

            text_embed_t = self.temporal.text_embed(text_tok.unsqueeze(1))
            audio_tok = self.depth.step(
                z, text_embed_t=text_embed_t,
                temperature=temperature, top_k=top_k_audio,
            )
            all_audio_tokens.append(audio_tok)

            prev_tgt_audio = audio_tok
            prev_text = text_tok

            if (text_tok == self.eos_id).all():
                break

        generated_audio = torch.stack(all_audio_tokens, dim=2)  # (B, Q, T_out)
        generated_text = torch.stack(all_text_tokens, dim=1)    # (B, T_out)

        # Remove acoustic delay before returning
        generated_audio = self.remove_acoustic_delay(generated_audio)

        return generated_audio, generated_text

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config: dict) -> "HibikiModel":
        """Create model from a config dict (parsed from YAML)."""
        model_cfg = config["model"]
        return cls(
            d_model=model_cfg["temporal"]["d_model"],
            ffn_dim=model_cfg["temporal"]["ffn_dim"],
            num_layers=model_cfg["temporal"]["num_layers"],
            num_heads=model_cfg["temporal"]["num_heads"],
            head_dim=model_cfg["temporal"]["head_dim"],
            max_seq_len=model_cfg["temporal"]["max_seq_len"],
            local_window=model_cfg["temporal"]["local_attn_window"],
            depth_d_model=model_cfg["depth"]["d_model"],
            depth_ffn_dim=model_cfg["depth"]["ffn_dim"],
            depth_layers_per_cb=model_cfg["depth"]["num_layers_per_codebook"],
            num_codebooks=model_cfg["codec"]["num_codebooks"],
            weight_sharing_start=model_cfg["depth"]["weight_sharing_start"],
            text_vocab_size=model_cfg["tokens"]["text_vocab_size"],
            audio_codebook_size=model_cfg["tokens"]["audio_codebook_size"],
            num_voice_categories=model_cfg["voice_transfer"]["num_categories"],
            acoustic_delay=model_cfg["tokens"]["acoustic_delay"],
            dropout=model_cfg["temporal"]["dropout"],
        )
