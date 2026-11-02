"""
audio_preprocessing.py
-----------------------
Phase 1 of the pipeline: turns a raw call-audio waveform into a set of
low-level signal representations that downstream feature extractors
(acoustic / behavioral) consume.

Implemented with Librosa. All operations are deterministic given a fixed
sample rate, so results are reproducible across runs.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import librosa
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "librosa is required for vishing_ai.preprocessing. "
        "Install with `pip install librosa soundfile`."
    ) from e

from vishing_ai.config import AUDIO_CFG, AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedAudio:
    """Container for all low-level signal representations of one call clip."""

    waveform: np.ndarray            # (n_samples,) mono, resampled, normalized
    sample_rate: int
    mfcc: np.ndarray                # (n_mfcc, T)
    stft_magnitude: np.ndarray      # (n_fft//2 + 1, T)
    f0_contour: np.ndarray          # (T,) fundamental frequency, NaN where unvoiced
    rms_energy: np.ndarray          # (T,)
    duration_sec: float
    source_path: Optional[str] = None

    @property
    def n_frames(self) -> int:
        return self.mfcc.shape[1]


class AudioPreprocessor:
    """
    Loads raw audio and computes the standard signal representations used
    throughout the pipeline: MFCCs, STFT magnitude spectrogram, F0 contour
    (via the YIN/pYIN algorithm), and frame-level RMS energy.

    Parameters
    ----------
    cfg : AudioConfig
        Audio configuration (sample rate, FFT/hop sizes, etc). Defaults to
        the project-wide config in `vishing_ai.config`.
    """

    def __init__(self, cfg: AudioConfig = AUDIO_CFG):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Loading & cleaning
    # ------------------------------------------------------------------
    def load(self, path: Union[str, Path]) -> tuple[np.ndarray, int]:
        """Load an audio file, resample to the target rate, force mono."""
        path = str(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path, sr=self.cfg.sample_rate, mono=True)
        return y, sr

    def trim_and_normalize(self, y: np.ndarray) -> np.ndarray:
        """Trim leading/trailing silence and peak-normalize amplitude."""
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        if y_trimmed.size == 0:
            y_trimmed = y  # degenerate case: entire clip was "silence"
        peak = np.max(np.abs(y_trimmed)) + 1e-8
        return (y_trimmed / peak).astype(np.float32)

    def enforce_duration(self, y: np.ndarray) -> np.ndarray:
        """Truncate overly long clips; leave short clips for caller to reject."""
        max_len = int(self.cfg.max_duration_sec * self.cfg.sample_rate)
        if len(y) > max_len:
            y = y[:max_len]
        return y

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------
    def compute_mfcc(self, y: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(
            y=y,
            sr=self.cfg.sample_rate,
            n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
        )

    def compute_stft_magnitude(self, y: np.ndarray) -> np.ndarray:
        stft = librosa.stft(
            y,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
        )
        return np.abs(stft)

    def compute_f0(self, y: np.ndarray) -> np.ndarray:
        """Fundamental frequency contour via pYIN. Unvoiced frames -> NaN."""
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            sr=self.cfg.sample_rate,
            hop_length=self.cfg.hop_length,
        )
        f0 = np.where(voiced_flag, f0, np.nan)
        return f0

    def compute_rms(self, y: np.ndarray) -> np.ndarray:
        return librosa.feature.rms(
            y=y, frame_length=self.cfg.win_length, hop_length=self.cfg.hop_length
        )[0]

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def process(self, path: Union[str, Path]) -> Optional[PreprocessedAudio]:
        """
        Full preprocessing pipeline for a single audio file.

        Returns None (and logs a warning) if the clip is too short to be
        usable, rather than raising, so batch pipelines can skip bad files.
        """
        y, sr = self.load(path)
        y = self.trim_and_normalize(y)
        duration = len(y) / sr

        if duration < self.cfg.min_duration_sec:
            logger.warning("Rejecting %s: duration %.2fs < min %.2fs",
                            path, duration, self.cfg.min_duration_sec)
            return None

        y = self.enforce_duration(y)

        mfcc = self.compute_mfcc(y)
        stft_mag = self.compute_stft_magnitude(y)
        f0 = self.compute_f0(y)
        rms = self.compute_rms(y)

        # Align lengths defensively (librosa framings can differ by 1 frame
        # depending on centering) — trim all to the shortest common length.
        min_T = min(mfcc.shape[1], stft_mag.shape[1], len(f0), len(rms))
        mfcc, stft_mag, f0, rms = mfcc[:, :min_T], stft_mag[:, :min_T], f0[:min_T], rms[:min_T]

        return PreprocessedAudio(
            waveform=y,
            sample_rate=sr,
            mfcc=mfcc,
            stft_magnitude=stft_mag,
            f0_contour=f0,
            rms_energy=rms,
            duration_sec=duration,
            source_path=str(path),
        )
