"""Temporal Transformer — the main backbone of the Hibiki model.

Operates at the codec frame rate (12.5Hz) and processes the full
sequence of tokens over time. Uses gated SiLU FFN, RoPE, and
local attention windowing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._cos_cache = None
        self._sin_cache = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos_cache is not None and self._cos_cache.shape[0] >= seq_len:
            return
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, offset: int = 0):
        """Returns (cos, sin) for positions [offset, offset+seq_len)."""
        seq_len = x.shape[-2]
        total = offset + seq_len
        self._build_cache(total, x.device, x.dtype)
        return (
            self._cos_cache[offset:total].unsqueeze(0).unsqueeze(0),
            self._sin_cache[offset:total].unsqueeze(0).unsqueeze(0),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class GatedSiLUFFN(nn.Module):
    """Gated SiLU Feed-Forward Network (as in LLaMA/Hibiki)."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TemporalAttention(nn.Module):
    """Multi-head self-attention with RoPE and optional local windowing."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.1,
        local_window: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.local_window = local_window
        self.scale = head_dim ** -0.5

        total_dim = num_heads * head_dim
        self.q_proj = nn.Linear(d_model, total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_dim, bias=False)
        self.o_proj = nn.Linear(total_dim, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin)

        # KV cache for autoregressive inference
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        # Use scaled dot-product attention (PyTorch 2.0+)
        # Causal mask is applied automatically with is_causal=True
        if kv_cache is not None:
            # During inference with cache, no causal mask needed (single step)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0 if not self.training else self.attn_dropout.p,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=0.0 if not self.training else self.attn_dropout.p,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out), new_cache


class TemporalTransformerLayer(nn.Module):
    """Single layer of the Temporal Transformer."""

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.1,
        local_window: Optional[int] = None,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = TemporalAttention(d_model, num_heads, head_dim, dropout, local_window)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = GatedSiLUFFN(d_model, ffn_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm attention
        residual = x
        x = self.attn_norm(x)
        x, new_cache = self.attn(x, cos, sin, kv_cache)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        return x, new_cache


class TemporalTransformer(nn.Module):
    """Temporal (Global) Transformer backbone.

    Processes sequences at the codec frame rate, modeling dependencies
    over time across all token streams.
    """

    def __init__(
        self,
        d_model: int = 512,
        ffn_dim: int = 1408,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64,
        max_seq_len: int = 250,
        local_window: Optional[int] = 250,
        dropout: float = 0.1,
        text_vocab_size: int = 32000,
        audio_codebook_size: int = 2048,
        num_codebooks: int = 8,
        num_streams: int = 2,  # source + target
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_codebooks = num_codebooks
        self.num_streams = num_streams

        # Token embeddings
        # Text stream embedding (for inner monologue)
        self.text_embed = nn.Embedding(text_vocab_size + 4, d_model)  # +4 for special tokens
        # Semantic token embeddings (codebook 1) per stream
        self.semantic_embeds = nn.ModuleList([
            nn.Embedding(audio_codebook_size + 4, d_model) for _ in range(num_streams)
        ])
        # Acoustic token embeddings (codebooks 2-Q) per stream — low-rank
        self.acoustic_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(audio_codebook_size + 4, 64),
                nn.Linear(64, d_model, bias=False),
            )
            for _ in range(num_streams * (num_codebooks - 1))
        ])

        # Positional encoding
        self.rope = RotaryEmbedding(head_dim, max_seq_len + 64)

        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(
                d_model, ffn_dim, num_heads, head_dim, dropout, local_window
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(d_model)

        # Output head for text tokens
        self.text_head = nn.Linear(d_model, text_vocab_size + 4, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed_tokens(
        self,
        text_tokens: Optional[torch.Tensor],
        target_audio_tokens: Optional[torch.Tensor],
        source_audio_tokens: Optional[torch.Tensor],
        voice_cond_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine all token embeddings into a single representation.

        Args:
            text_tokens: (B, T) text token indices
            target_audio_tokens: (B, Q, T) target stream audio tokens
            source_audio_tokens: (B, Q, T) source stream audio tokens
            voice_cond_embed: (B, T, D) optional voice conditioning embedding

        Returns:
            (B, T, d_model) combined embedding
        """
        B = text_tokens.shape[0] if text_tokens is not None else target_audio_tokens.shape[0]
        T = text_tokens.shape[1] if text_tokens is not None else target_audio_tokens.shape[2]

        h = torch.zeros(B, T, self.d_model, device=text_tokens.device if text_tokens is not None else target_audio_tokens.device)

        # Text embedding
        if text_tokens is not None:
            h = h + self.text_embed(text_tokens)

        # Target stream embeddings
        if target_audio_tokens is not None:
            # Semantic token (codebook 0)
            h = h + self.semantic_embeds[0](target_audio_tokens[:, 0, :])
            # Acoustic tokens (codebooks 1 to Q-1)
            for q in range(1, min(target_audio_tokens.shape[1], self.num_codebooks)):
                embed_idx = q - 1  # target stream acoustic embeds
                h = h + self.acoustic_embeds[embed_idx](target_audio_tokens[:, q, :])

        # Source stream embeddings
        if source_audio_tokens is not None:
            # Semantic token
            h = h + self.semantic_embeds[1](source_audio_tokens[:, 0, :])
            # Acoustic tokens
            offset = self.num_codebooks - 1  # offset for source stream embeds
            for q in range(1, min(source_audio_tokens.shape[1], self.num_codebooks)):
                embed_idx = offset + (q - 1)
                h = h + self.acoustic_embeds[embed_idx](source_audio_tokens[:, q, :])

        # Voice conditioning
        if voice_cond_embed is not None:
            h = h + voice_cond_embed

        return h

    def forward(
        self,
        h: torch.Tensor,
        offset: int = 0,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Forward pass through the temporal transformer.

        Args:
            h: (B, T, d_model) input embeddings
            offset: position offset for RoPE (used in autoregressive inference)
            kv_caches: list of (K, V) tuples per layer for cached inference

        Returns:
            z: (B, T, d_model) output representations
            text_logits: (B, T, text_vocab_size) logits for text prediction
            new_kv_caches: updated KV caches
        """
        cos, sin = self.rope(h, offset=offset)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = layer(h, cos, sin, cache)
            new_kv_caches.append(new_cache)

        z = self.final_norm(h)
        text_logits = self.text_head(z)

        return z, text_logits, new_kv_caches

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
