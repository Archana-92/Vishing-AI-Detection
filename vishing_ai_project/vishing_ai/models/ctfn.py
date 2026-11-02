"""
ctfn.py
-------
Convolutional Transformer Fusion Network (CTFN) — the core model of this
project.

Architecture summary
---------------------
1. Acoustic branch: 1D-CNN stack (local temporal pattern extraction, e.g.
   phoneme-scale energy/pitch transients) -> multi-head Transformer encoder
   (long-range dependencies across the utterance, e.g. prosodic patterns
   spanning the whole call) -> mean-pooled embedding.
2. Linguistic branch: MLP over the transcript's bag-of-tokens embedding and
   scam-keyword vector.
3. Behavioral branch: MLP over the fixed-size behavioral statistics vector.
4. Cross-modal attention fusion combines the three branch embeddings into a
   shared fusion embedding, with attention weights exposed for XAI.
5. Classification head: MLP -> softmax over {scam, legitimate, casual}.

All nonlinearities use TrainableSwish (per-channel learnable beta) rather
than fixed ReLU/GELU, per the architecture design in the project spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from vishing_ai.config import FEATURE_CFG, MODEL_CFG, ModelConfig, FeatureConfig
from vishing_ai.models.layers import TrainableSwish, PositionalEncoding, CrossModalAttentionFusion
from vishing_ai.features.linguistic import _ALL_KEYWORDS

NUM_KEYWORDS = len(_ALL_KEYWORDS)


@dataclass
class CTFNOutput:
    logits: torch.Tensor                 # (B, num_classes)
    probs: torch.Tensor                  # (B, num_classes)
    modality_weights: torch.Tensor       # (B, 3) [acoustic, linguistic, behavioral]
    temporal_attention: Optional[torch.Tensor]  # (B, T) self-attn mass per acoustic frame
    fused_embedding: torch.Tensor        # (B, fusion_dim)


class AcousticEncoder(nn.Module):
    """1D-CNN -> Transformer encoder over the frame-level acoustic sequence."""

    def __init__(self, cfg: ModelConfig, in_dim: int, seq_len: int):
        super().__init__()
        channels = [in_dim] + cfg.cnn_channels
        convs = []
        for i in range(len(cfg.cnn_channels)):
            convs.append(nn.Conv1d(channels[i], channels[i + 1],
                                    kernel_size=cfg.cnn_kernel_sizes[i],
                                    padding=cfg.cnn_kernel_sizes[i] // 2))
            convs.append(nn.BatchNorm1d(channels[i + 1]))
            convs.append(TrainableSwish(channels[i + 1], channel_dim=1))
            convs.append(nn.Dropout(cfg.dropout))
        self.cnn = nn.Sequential(*convs)

        d_model = cfg.cnn_channels[-1]
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        self.out_dim = d_model

        # learned temporal attention pooling (also used for XAI saliency)
        self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (B, T, in_dim), mask: (B, T) bool, True = pad (ignore).
        Returns pooled embedding (B, d_model) and per-frame attention (B, T).
        """
        x = x.transpose(1, 2)          # (B, in_dim, T)
        x = self.cnn(x)                # (B, d_model, T)
        x = x.transpose(1, 2)          # (B, T, d_model)
        x = self.pos_enc(x)

        x = self.transformer(x, src_key_padding_mask=mask)

        attn_logits = self.attn_pool(x).squeeze(-1)  # (B, T)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, T)

        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, d_model)
        return pooled, attn_weights


class BranchMLP(nn.Module):
    """Small MLP used for the linguistic and behavioral branches."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            TrainableSwish(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            TrainableSwish(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CTFN(nn.Module):
    """Full Convolutional Transformer Fusion Network."""

    def __init__(self, model_cfg: ModelConfig = MODEL_CFG, feature_cfg: FeatureConfig = FEATURE_CFG):
        super().__init__()
        self.model_cfg = model_cfg
        self.feature_cfg = feature_cfg

        self.acoustic_encoder = AcousticEncoder(
            model_cfg, in_dim=feature_cfg.acoustic_dim, seq_len=feature_cfg.max_seq_len
        )

        linguistic_in = feature_cfg.linguistic_embed_dim + NUM_KEYWORDS
        self.linguistic_branch = BranchMLP(
            in_dim=linguistic_in, hidden_dim=128, out_dim=128, dropout=model_cfg.dropout
        )

        self.behavioral_branch = BranchMLP(
            in_dim=feature_cfg.behavioral_dim, hidden_dim=32, out_dim=32, dropout=model_cfg.dropout
        )

        self.fusion = CrossModalAttentionFusion(
            acoustic_dim=self.acoustic_encoder.out_dim,
            linguistic_dim=128,
            behavioral_dim=32,
            fusion_dim=feature_cfg.fusion_dim,
            num_heads=model_cfg.attention_fusion_heads,
            dropout=model_cfg.dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_cfg.fusion_dim, feature_cfg.fusion_dim // 2),
            TrainableSwish(feature_cfg.fusion_dim // 2),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(feature_cfg.fusion_dim // 2, model_cfg.num_classes),
        )

    def forward(self, acoustic_seq: torch.Tensor, linguistic_embedding: torch.Tensor,
                keyword_vector: torch.Tensor, behavioral_vector: torch.Tensor,
                acoustic_mask: Optional[torch.Tensor] = None) -> CTFNOutput:
        acoustic_pooled, temporal_attn = self.acoustic_encoder(acoustic_seq, acoustic_mask)

        linguistic_in = torch.cat([linguistic_embedding, keyword_vector], dim=-1)
        linguistic_emb = self.linguistic_branch(linguistic_in)

        behavioral_emb = self.behavioral_branch(behavioral_vector)

        fused, modality_weights = self.fusion(acoustic_pooled, linguistic_emb, behavioral_emb)

        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)

        return CTFNOutput(
            logits=logits,
            probs=probs,
            modality_weights=modality_weights,
            temporal_attention=temporal_attn,
            fused_embedding=fused,
        )
