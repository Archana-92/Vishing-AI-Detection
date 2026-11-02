"""
explainer.py
------------
Phase 4 of the pipeline: turns a CTFN forward pass (attention weights over
modalities and over time) plus the underlying signal-level features into a
human-readable rationale string, so predictions are auditable rather than
opaque probabilities.

Design note: this is intentionally a *post-hoc, attention-based* explainer
rather than a separate surrogate model (e.g. LIME/SHAP). Attention weights
are not a perfect proxy for causal feature importance — this is disclosed
in README.md's limitations section — but combined with concrete signal
checks (e.g. "3 urgency-related phrases detected") they give a reasonably
faithful, cheap-to-compute account of what drove a prediction, appropriate
as a first XAI layer for a research prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from vishing_ai.config import IDX2LABEL
from vishing_ai.features.extractor import CallFeatures
from vishing_ai.features.linguistic import SCAM_KEYWORD_GROUPS
from vishing_ai.models.ctfn import CTFNOutput

MODALITY_NAMES = ["acoustic", "linguistic", "behavioral"]


@dataclass
class Explanation:
    predicted_label: str
    confidence: float
    modality_contributions: dict            # {"acoustic": 0.41, ...}
    top_keyword_matches: List[str]
    high_attention_time_fraction: float      # fraction of call driving most attention
    behavioral_flags: List[str] = field(default_factory=list)
    rationale: str = ""

    def __str__(self) -> str:
        return self.rationale


class CTFNExplainer:
    """Builds an `Explanation` from a single-example CTFN output + features."""

    def __init__(self, behavioral_thresholds: dict | None = None):
        # Thresholds chosen as descriptive heuristics for flagging unusual
        # behavioral statistics; NOT statistically validated cutoffs — stated
        # explicitly here and in README as a disclosed limitation.
        self.thresholds = behavioral_thresholds or {
            "speech_rate_high_wps": 3.3,
            "hesitation_index_low": 1.0,
            "pause_count_low": 2,
        }

    def explain(self, output: CTFNOutput, features: CallFeatures, example_idx: int = 0) -> Explanation:
        probs = output.probs[example_idx].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = IDX2LABEL[pred_idx]
        confidence = float(probs[pred_idx])

        mod_w = output.modality_weights[example_idx].detach().cpu().numpy()
        modality_contributions = {name: float(w) for name, w in zip(MODALITY_NAMES, mod_w)}

        if output.temporal_attention is not None:
            temporal = output.temporal_attention[example_idx].detach().cpu().numpy()
            # fraction of frames responsible for the top 50% of attention mass
            sorted_attn = np.sort(temporal)[::-1]
            cumsum = np.cumsum(sorted_attn)
            half_point = np.searchsorted(cumsum, 0.5 * cumsum[-1]) + 1
            high_attention_fraction = float(half_point / max(len(temporal), 1))
        else:
            high_attention_fraction = float("nan")

        behavioral_flags = self._flag_behavioral(features)

        rationale = self._build_rationale(
            pred_label, confidence, modality_contributions,
            features.matched_keywords, behavioral_flags,
        )

        return Explanation(
            predicted_label=pred_label,
            confidence=confidence,
            modality_contributions=modality_contributions,
            top_keyword_matches=features.matched_keywords[:5],
            high_attention_time_fraction=high_attention_fraction,
            behavioral_flags=behavioral_flags,
            rationale=rationale,
        )

    def _flag_behavioral(self, features: CallFeatures) -> List[str]:
        flags = []
        speech_rate, pause_count, hesitation_idx, _, _, _ = features.behavioral_vector
        if speech_rate > self.thresholds["speech_rate_high_wps"]:
            flags.append(f"unusually fast speech rate ({speech_rate:.1f} words/sec)")
        if hesitation_idx < self.thresholds["hesitation_index_low"]:
            flags.append("low natural hesitation (script-like delivery)")
        if pause_count < self.thresholds["pause_count_low"]:
            flags.append("few natural pauses (possible scripted/read speech)")
        return flags

    def _keyword_group_summary(self, matched: List[str]) -> List[str]:
        summary = []
        for group, kws in SCAM_KEYWORD_GROUPS.items():
            hits = [m for m in matched if m in kws]
            if hits:
                summary.append(f"{group.replace('_', ' ')} ({len(hits)} phrase(s): "
                                f"{', '.join(hits[:3])}{'...' if len(hits) > 3 else ''})")
        return summary

    def _build_rationale(self, label: str, confidence: float, modality_contributions: dict,
                          matched_keywords: List[str], behavioral_flags: List[str]) -> str:
        dominant_modality = max(modality_contributions, key=modality_contributions.get)
        parts = [
            f"Predicted '{label}' with {confidence:.1%} confidence.",
            f"The {dominant_modality} signal contributed most "
            f"({modality_contributions[dominant_modality]:.1%} attention weight).",
        ]

        kw_summary = self._keyword_group_summary(matched_keywords)
        if kw_summary:
            parts.append("Linguistic red flags detected: " + "; ".join(kw_summary) + ".")
        elif label == "scam":
            parts.append("No lexical scam markers matched; prediction driven by "
                          "acoustic/behavioral signal rather than keyword content.")

        if behavioral_flags:
            parts.append("Behavioral indicators: " + "; ".join(behavioral_flags) + ".")

        return " ".join(parts)
