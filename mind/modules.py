# -*- coding: utf-8 -*-
"""
Building blocks for AFIRE / MIND.

Uses ``x_transformers`` for the Transformer backbone (same as the original
standalone code) to preserve RoPE, ScaleNorm, and scaled residuals that are
critical for performance.
"""
from __future__ import annotations

from torch import nn


# --------------------------------------------------------------------------- #
#  MLP projector
# --------------------------------------------------------------------------- #
def build_projector(in_dim: int, out_dim: int) -> nn.Module:
    """Linear projector (matches original MlpConfig with no hidden_sizes)."""
    return nn.Linear(in_dim, out_dim)


# --------------------------------------------------------------------------- #
#  Transformer Encoder  (wraps x_transformers.Encoder)
# --------------------------------------------------------------------------- #
def build_transformer_encoder(
    dim: int,
    depth: int = 8,
    heads: int = 8,
    attn_dropout: float = 0.0,
    ff_dropout: float = 0.0,
    ff_mult: int = 4,
) -> nn.Module:
    """Build an x_transformers Encoder with the same defaults as the original
    ``TransformerEncoderConfig`` used in standalone/weighted_moe_decoder.py.

    Key features preserved from the original:
      - use_scalenorm=True   (ScaleNorm instead of LayerNorm)
      - rotary_pos_emb=True  (Rotary Position Embeddings)
      - scale_residual=True  (scaled residual connections)
    """
    from x_transformers import Encoder

    attn_dim_head = dim // heads
    return Encoder(
        dim=dim,
        depth=depth,
        heads=heads,
        attn_dim_head=attn_dim_head,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        ff_mult=ff_mult,
        layer_dropout=0.0,
        use_scalenorm=True,
        use_rmsnorm=False,
        rotary_pos_emb=True,
        rotary_xpos=False,
        rel_pos_bias=False,
        alibi_pos_bias=False,
        residual_attn=False,
        scale_residual=True,
        cross_attend=False,
        attn_flash=False,
    )
