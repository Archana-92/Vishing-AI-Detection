"""
behavioral.py
-------------
Derives call-level behavioral / paralinguistic statistics that are known
correlates of scripted, high-pressure speech: speech rate, pause structure,
hesitation markers, and energy dynamics. These are computed from the same
preprocessed audio used by the acoustic branch, plus the transcript for
speech-rate normalization and hesitation-token counting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from vishing_ai.preprocessing.audio_preprocessing import PreprocessedAudio

HESITATION_MARKERS = {"um", "uh", "er", "erm", "ah", "hmm", "like", "you know"}


@dataclass
class BehavioralResult:
    speech_rate_wps: float          # words per second
    pause_count: int                # number of silent gaps > 250ms
    hesitation_index: float         # hesitation markers per 100 words
    energy_variance: float          # variance of RMS energy across the call
    pitch_jitter: float             # mean absolute frame-to-frame F0 change
    talk_silence_ratio: float       # fraction of frames above energy threshold

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.speech_rate_wps,
            self.pause_count,
            self.hesitation_index,
            self.energy_variance,
            self.pitch_jitter,
            self.talk_silence_ratio,
        ], dtype=np.float32)


def _count_pauses(rms: np.ndarray, hop_length: int, sample_rate: int,
                   silence_thresh_ratio: float = 0.1, min_pause_sec: float = 0.25) -> int:
    thresh = silence_thresh_ratio * (np.max(rms) + 1e-8)
    silent = rms < thresh
    min_frames = int(min_pause_sec * sample_rate / hop_length)

    pauses, run = 0, 0
    for s in silent:
        if s:
            run += 1
        else:
            if run >= min_frames:
                pauses += 1
            run = 0
    if run >= min_frames:
        pauses += 1
    return pauses


def _hesitation_count(text: str) -> int:
    tokens = re.findall(r"[a-z']+", text.lower())
    return sum(1 for t in tokens if t in HESITATION_MARKERS)


def extract_behavioral_features(audio: PreprocessedAudio, transcript: str,
                                 hop_length: int) -> BehavioralResult:
    words = re.findall(r"[a-zA-Z']+", transcript)
    n_words = max(len(words), 1)
    duration = max(audio.duration_sec, 1e-3)

    speech_rate = n_words / duration
    pause_count = _count_pauses(audio.rms_energy, hop_length, audio.sample_rate)
    hesitations = _hesitation_count(transcript)
    hesitation_index = 100.0 * hesitations / n_words

    energy_variance = float(np.var(audio.rms_energy))

    voiced_f0 = audio.f0_contour[~np.isnan(audio.f0_contour)]
    if len(voiced_f0) > 1:
        pitch_jitter = float(np.mean(np.abs(np.diff(voiced_f0))))
    else:
        pitch_jitter = 0.0

    energy_thresh = 0.1 * (np.max(audio.rms_energy) + 1e-8)
    talk_silence_ratio = float(np.mean(audio.rms_energy >= energy_thresh))

    return BehavioralResult(
        speech_rate_wps=speech_rate,
        pause_count=pause_count,
        hesitation_index=hesitation_index,
        energy_variance=energy_variance,
        pitch_jitter=pitch_jitter,
        talk_silence_ratio=talk_silence_ratio,
    )
