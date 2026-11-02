"""
config.py
---------
Single source of truth for hyperparameters, paths, and constants used
across preprocessing, feature extraction, modeling, training, and inference.

Centralizing configuration here (rather than scattering magic numbers through
the codebase) is standard practice for reproducible research code and makes
it trivial to report the exact configuration used for any reported result.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
RAW_AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
METADATA_CSV = DATA_DIR / "metadata.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (CHECKPOINT_DIR, LOG_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Class labels
# --------------------------------------------------------------------------
LABELS: List[str] = ["scam", "legitimate", "casual"]
LABEL2IDX = {label: i for i, label in enumerate(LABELS)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}
NUM_CLASSES = len(LABELS)


# --------------------------------------------------------------------------
# Audio preprocessing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    n_fft: int = 512
    hop_length: int = 160          # 10ms at 16kHz
    win_length: int = 400          # 25ms at 16kHz
    n_mfcc: int = 40
    n_mels: int = 80
    fmin: float = 50.0
    fmax: float = 7600.0
    max_duration_sec: float = 30.0  # clips longer than this are truncated
    min_duration_sec: float = 1.5   # clips shorter than this are rejected


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    # acoustic branch (frame-level, sequence length = T)
    acoustic_dim: int = 40 + 1 + 1 + (513 // 4 + 1)   # mfcc + f0 + rms + pooled STFT bins
    max_seq_len: int = 300               # frames, ~3s context window after pooling

    # linguistic branch
    whisper_model_size: str = "base"
    max_tokens: int = 128
    vocab_size: int = 8000               # bag-of-tokens / learned embedding vocab
    linguistic_embed_dim: int = 128

    # behavioral branch (fixed-size statistical summary per call)
    behavioral_dim: int = 6              # speech rate, pause count, hesitation idx,
                                          # energy variance, mean pitch jitter, talk/silence ratio

    # shared fusion space
    fusion_dim: int = 256


# --------------------------------------------------------------------------
# Model (CTFN) hyperparameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [5, 5, 3])
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff_dim: int = 512
    dropout: float = 0.2
    swish_beta_init: float = 1.0         # trainable Swish beta initial value
    attention_fusion_heads: int = 4
    num_classes: int = NUM_CLASSES


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    k_folds: int = 5
    seed: int = 42
    early_stopping_patience: int = 6
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    device: str = "cuda"  # falls back to cpu automatically at runtime


AUDIO_CFG = AudioConfig()
FEATURE_CFG = FeatureConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
