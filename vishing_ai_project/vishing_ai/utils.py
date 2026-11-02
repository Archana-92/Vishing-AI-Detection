"""
utils.py
--------
Shared utilities: reproducibility helpers, the PyTorch Dataset wrapping
cached `CallFeatures`, and metric computation used by train.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from vishing_ai.config import LABEL2IDX
from vishing_ai.features.extractor import CallFeatures


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class LabeledCallFeatures:
    features: CallFeatures
    label: str


class VishingDataset(Dataset):
    """
    Wraps a list of `LabeledCallFeatures` for PyTorch training. Feature
    extraction is assumed to have been done ahead of time (e.g. cached to
    disk) since ASR + audio feature extraction is comparatively expensive;
    this keeps the training loop itself fast and simple.
    """

    def __init__(self, items: Sequence[LabeledCallFeatures]):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        f = item.features
        return {
            "acoustic_seq": torch.from_numpy(f.acoustic_seq).float(),
            "linguistic_embedding": torch.from_numpy(f.linguistic_embedding).float(),
            "keyword_vector": torch.from_numpy(f.keyword_vector).float(),
            "behavioral_vector": torch.from_numpy(f.behavioral_vector).float(),
            "label": torch.tensor(LABEL2IDX[item.label], dtype=torch.long),
        }


def collate_batch(batch: List[dict]) -> dict:
    return {
        "acoustic_seq": torch.stack([b["acoustic_seq"] for b in batch]),
        "linguistic_embedding": torch.stack([b["linguistic_embedding"] for b in batch]),
        "keyword_vector": torch.stack([b["keyword_vector"] for b in batch]),
        "behavioral_vector": torch.stack([b["behavioral_vector"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=-1)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))
