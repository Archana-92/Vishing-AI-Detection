"""
layers.py
---------
Reusable building blocks for the Convolutional Transformer Fusion Network
(CTFN): a trainable-beta Swish activation and a cross-modal attention
fusion layer that projects acoustic, linguistic, and behavioral branch
embeddings into a shared space and combines them via learned attention
weights (rather than naive concatenation), which is also the hook the XAI
module reads modality-importance from.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainableSwish(nn.Module):
    """
    Swish activation, x * sigmoid(beta * x), with a learnable per-channel
    (or scalar) beta parameter. Beta=1 recovers standard SiLU; beta -> large
    approaches ReLU; beta -> 0 approaches a linear unit. Learning beta lets
    each layer adapt its own nonlinearity sharpness during training.
    """

    def __init__(self, num_features: Optional[int] = None, beta_init: float = 1.0,
                 channel_dim: int = -1):
        """
        Parameters
        ----------
        num_features : per-channel beta size, or None for a single scalar beta.
        channel_dim : which dimension of the input holds the channel axis
            that `num_features` indexes. Use -1 (default) for (B, ..., C)
            tensors (e.g. after Linear layers) and 1 for (B, C, T) tensors
            (e.g. after Conv1d layers).
        """
        super().__init__()
        shape = (1,) if num_features is None else (num_features,)
        self.beta = nn.Parameter(torch.full(shape, beta_init, dtype=torch.float32))
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.beta.numel() == 1:
            beta = self.beta
        else:
            # build a broadcast shape with beta's values on `channel_dim`
            view_shape = [1] * x.dim()
            view_shape[self.channel_dim] = -1
            beta = self.beta.view(*view_shape)
        return x * torch.sigmoid(beta * x)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for the Transformer encoder."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class CrossModalAttentionFusion(nn.Module):
    """
    Projects each modality's pooled embedding into a shared fusion space,
    then computes attention weights over the three modalities (query =
    mean of the three projections) to produce a single fused embedding as
    a weighted combination. This is the layer whose attention weights the
    XAI module surfaces as "modality contribution" in its explanations.
    """

    def __init__(self, acoustic_dim: int, linguistic_dim: int, behavioral_dim: int,
                 fusion_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.acoustic_proj = nn.Linear(acoustic_dim, fusion_dim)
        self.linguistic_proj = nn.Linear(linguistic_dim, fusion_dim)
        self.behavioral_proj = nn.Linear(behavioral_dim, fusion_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

        self.fusion_dim = fusion_dim

    def forward(self, acoustic_emb: torch.Tensor, linguistic_emb: torch.Tensor,
                behavioral_emb: torch.Tensor):
        """
        Each input: (batch, raw_dim). Returns:
          fused:            (batch, fusion_dim)
          modality_weights: (batch, 3) attention mass assigned to
                             [acoustic, linguistic, behavioral] — used by XAI.
        """
        a = self.acoustic_proj(acoustic_emb)      # (B, F)
        l = self.linguistic_proj(linguistic_emb)  # (B, F)
        b = self.behavioral_proj(behavioral_emb)  # (B, F)

        modalities = torch.stack([a, l, b], dim=1)   # (B, 3, F)
        query = modalities.mean(dim=1, keepdim=True)  # (B, 1, F)

        attn_out, attn_weights = self.mha(query, modalities, modalities, need_weights=True)
        # attn_weights: (B, 1, 3) -> (B, 3)
        modality_weights = attn_weights.squeeze(1)

        fused = self.norm(attn_out.squeeze(1) + query.squeeze(1))
        fused = self.dropout(fused)
        return fused, modality_weights
