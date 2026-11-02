"""
acoustic.py
-----------
Builds the frame-level acoustic feature sequence fed to the CNN+Transformer
branch of the CTFN model: MFCCs, F0 contour, RMS energy, and a pooled STFT
magnitude summary, concatenated per frame and padded/truncated to a fixed
sequence length.
"""

from __future__ import annotations

import numpy as np

from vishing_ai.config import AUDIO_CFG, FEATURE_CFG
from vishing_ai.preprocessing.audio_preprocessing import PreprocessedAudio


def _pool_stft(stft_mag: np.ndarray, target_bins: int) -> np.ndarray:
    """Average-pool the frequency axis of a magnitude spectrogram down to
    `target_bins` bands so it can be concatenated with MFCCs without
    dominating the feature vector dimensionality."""
    n_bins, T = stft_mag.shape
    pool_factor = max(1, n_bins // target_bins)
    trimmed = stft_mag[: pool_factor * target_bins, :]
    pooled = trimmed.reshape(target_bins, pool_factor, T).mean(axis=1)
    return pooled


def _impute_f0(f0: np.ndarray) -> np.ndarray:
    """Replace NaN (unvoiced) frames with 0 after z-scoring voiced frames,
    a standard, simple imputation that avoids leaking NaNs into the model."""
    voiced_mask = ~np.isnan(f0)
    if voiced_mask.sum() == 0:
        return np.zeros_like(f0)
    mean = f0[voiced_mask].mean()
    std = f0[voiced_mask].std() + 1e-6
    f0_norm = (f0 - mean) / std
    f0_norm[~voiced_mask] = 0.0
    return f0_norm


def extract_acoustic_sequence(audio: PreprocessedAudio) -> np.ndarray:
    """
    Returns
    -------
    np.ndarray, shape (max_seq_len, acoustic_dim)
        Frame-level acoustic feature sequence, zero-padded or truncated to
        `FEATURE_CFG.max_seq_len` frames.
    """
    mfcc = audio.mfcc.T                                  # (T, n_mfcc)
    f0 = _impute_f0(audio.f0_contour)[:, None]            # (T, 1)
    rms = audio.rms_energy[:, None]                        # (T, 1)

    stft_bins = FEATURE_CFG.acoustic_dim - mfcc.shape[1] - 2
    stft_pooled = _pool_stft(audio.stft_magnitude, stft_bins).T  # (T, stft_bins)
    stft_pooled = np.log1p(stft_pooled)  # log-magnitude compresses dynamic range

    feats = np.concatenate([mfcc, f0, rms, stft_pooled], axis=1)  # (T, acoustic_dim)

    # per-utterance standardization (zero mean, unit variance per channel)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    feats = (feats - mean) / std

    return _pad_or_truncate(feats, FEATURE_CFG.max_seq_len)


def _pad_or_truncate(feats: np.ndarray, target_len: int) -> np.ndarray:
    T, D = feats.shape
    if T >= target_len:
        return feats[:target_len]
    pad = np.zeros((target_len - T, D), dtype=feats.dtype)
    return np.concatenate([feats, pad], axis=0)
