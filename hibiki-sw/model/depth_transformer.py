"""Depth Transformer — models the hierarchy of audio codebook tokens.

For each time step, the Depth Transformer autoregressively generates
tokens across codebook levels (q=1..Q), conditioned on the Temporal
Transformer's output representation Z_t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DepthTransformerLayer(nn.Module):
    """Single layer of the Depth Transformer."""

    def __init__(self, d_model: int, ffn_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True, bias=False,
        )
        self.ffn_norm = nn.RMSNorm(d_model)
        # Gated SiLU FFN
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x_norm = self.attn_norm(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)
        x = residual + x_attn

        # Pre-norm gated SiLU FFN
        residual = x
        x_norm = self.ffn_norm(x)
        x = residual + self.dropout(
            self.down_proj(F.silu(self.gate_proj(x_norm)) * self.up_proj(x_norm))
        )
        return x


class DepthBlock(nn.Module):
    """A stack of Depth Transformer layers for one or more codebook levels."""

    def __init__(self, d_model: int, ffn_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DepthTransformerLayer(d_model, ffn_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, causal_mask)
        return x


class DepthTransformer(nn.Module):
    """Depth Transformer for hierarchical audio token generation.

    For each time step t, takes the Temporal Transformer output Z_t
    and autoregressively generates tokens across codebook levels.

    The first Q steps generate output stream tokens, the next Q steps
    generate input stream tokens (training only).

    Weight sharing: codebooks 1-4 have independent weights,
    codebooks 5-8 share a single weight set.
    """

    def __init__(
        self,
        d_model: int = 384,
        ffn_dim: int = 1024,
        num_layers_per_codebook: int = 4,
        num_codebooks: int = 8,
        num_streams: int = 2,
        weight_sharing_start: int = 5,
        audio_codebook_size: int = 2048,
        embedding_dim: int = 64,
        temporal_d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.num_streams = num_streams
        self.weight_sharing_start = weight_sharing_start
        self.audio_codebook_size = audio_codebook_size
        num_heads = d_model // 64  # 384 // 64 = 6 heads

        # Project from temporal transformer dim to depth dim
        self.input_proj = nn.Linear(temporal_d_model, d_model, bias=False)

        # Token embeddings (low-rank: codebook_size -> embedding_dim -> d_model)
        self.token_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(audio_codebook_size + 4, embedding_dim),
                nn.Linear(embedding_dim, d_model, bias=False),
            )
            for _ in range(num_streams * num_codebooks)
        ])

        # Special start token embedding (for q=0 step, uses text token info)
        self.start_embed = nn.Linear(temporal_d_model, d_model, bias=False)

        # Depth blocks with weight sharing
        # Codebooks 1 to (sharing_start-1): independent blocks
        # Codebooks sharing_start to Q: shared block
        num_independent = min(weight_sharing_start - 1, num_codebooks)
        self.independent_blocks = nn.ModuleList([
            DepthBlock(d_model, ffn_dim, num_heads, num_layers_per_codebook, dropout)
            for _ in range(num_independent)
        ])
        # Shared block for remaining codebooks
        if num_codebooks >= weight_sharing_start:
            self.shared_block = DepthBlock(d_model, ffn_dim, num_heads, num_layers_per_codebook, dropout)
        else:
            self.shared_block = None

        # Output norms and heads per codebook level
        self.output_norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(num_codebooks)])
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, audio_codebook_size + 4, bias=False)
            for _ in range(num_codebooks)
        ])

        # Same for input stream (training only)
        self.input_output_norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(num_codebooks)])
        self.input_output_heads = nn.ModuleList([
            nn.Linear(d_model, audio_codebook_size + 4, bias=False)
            for _ in range(num_codebooks)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_block(self, codebook_idx: int) -> DepthBlock:
        """Get the depth block for a given codebook index (0-indexed)."""
        if codebook_idx < len(self.independent_blocks):
            return self.independent_blocks[codebook_idx]
        return self.shared_block

    def forward(
        self,
        z: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        source_tokens: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass — generates logits for all codebook levels.

        During training, uses teacher forcing with target_tokens.
        During inference, call step() for autoregressive generation.

        Args:
            z: (B, T, temporal_d_model) output from Temporal Transformer
            target_tokens: (B, Q, T) ground truth target audio tokens (training)
            source_tokens: (B, Q, T) ground truth source audio tokens (training)
            text_embed: (B, T, temporal_d_model) text token embedding for q=0 conditioning

        Returns:
            target_logits: (B, Q, T, codebook_size) logits for target stream
            source_logits: (B, Q, T, codebook_size) or None for source stream
        """
        B, T, _ = z.shape
        Q = self.num_codebooks

        z_proj = self.input_proj(z)  # (B, T, d_model)

        target_logits = []
        source_logits = []

        # Process each time step in parallel (teacher forcing)
        # For each codebook level q:
        for q in range(Q):
            # Build input sequence for depth transformer at level q
            if q == 0:
                # First codebook: conditioned on Z_t + text embedding
                if text_embed is not None:
                    h = z_proj + self.start_embed(text_embed)
                else:
                    h = z_proj
            else:
                # Subsequent codebooks: add previous target token embedding
                prev_tok = target_tokens[:, q - 1, :]  # (B, T)
                h = z_proj + self.token_embeds[q - 1](prev_tok)

            # Apply appropriate depth block
            block = self._get_block(q)
            h = block(h.unsqueeze(2)).squeeze(2) if h.dim() == 2 else block(h)

            # Output logits for target stream
            h_norm = self.output_norms[q](h)
            logits_q = self.output_heads[q](h_norm)
            target_logits.append(logits_q)

        # Source stream (training only)
        if source_tokens is not None:
            for q in range(Q):
                if q == 0:
                    # Conditioned on last target token embedding
                    last_target_tok = target_tokens[:, Q - 1, :]
                    h = z_proj + self.token_embeds[Q - 1](last_target_tok)
                else:
                    prev_tok = source_tokens[:, q - 1, :]
                    embed_idx = Q + (q - 1)
                    h = z_proj + self.token_embeds[embed_idx](prev_tok)

                block = self._get_block(q)
                h = block(h)

                h_norm = self.input_output_norms[q](h)
                logits_q = self.input_output_heads[q](h_norm)
                source_logits.append(logits_q)

        # Stack: (B, Q, T, vocab)
        target_logits = torch.stack(target_logits, dim=1)
        source_logits = torch.stack(source_logits, dim=1) if source_logits else None

        return target_logits, source_logits

    @torch.no_grad()
    def step(
        self,
        z_t: torch.Tensor,
        text_embed_t: Optional[torch.Tensor] = None,
        temperature: float = 0.8,
        top_k: int = 250,
    ) -> torch.Tensor:
        """Autoregressive generation for a single time step.

        Args:
            z_t: (B, 1, temporal_d_model) temporal output at time t
            text_embed_t: (B, 1, temporal_d_model) text embedding at time t
            temperature: sampling temperature
            top_k: top-k filtering

        Returns:
            tokens: (B, Q) generated audio tokens for this time step
        """
        B = z_t.shape[0]
        Q = self.num_codebooks
        z_proj = self.input_proj(z_t.squeeze(1))  # (B, d_model)

        tokens = []
        for q in range(Q):
            if q == 0:
                if text_embed_t is not None:
                    h = z_proj + self.start_embed(text_embed_t.squeeze(1))
                else:
                    h = z_proj
            else:
                prev_tok = tokens[-1]  # (B,)
                h = z_proj + self.token_embeds[q - 1](prev_tok)

            # Expand for block processing
            block = self._get_block(q)
            h = block(h.unsqueeze(1)).squeeze(1)  # (B, d_model)

            h_norm = self.output_norms[q](h)
            logits = self.output_heads[q](h_norm)  # (B, vocab)

            # Temperature + top-k sampling
            logits = logits / temperature
            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                logits[logits < topk_vals[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
            tokens.append(tok)

        return torch.stack(tokens, dim=1)  # (B, Q)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
