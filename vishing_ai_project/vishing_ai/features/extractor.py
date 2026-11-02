"""
extractor.py
------------
Phase 2 of the pipeline: orchestrates acoustic, linguistic, and behavioral
feature extraction into a single `CallFeatures` bundle ready for the CTFN
model (or for caching to disk ahead of training).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from vishing_ai.config import AUDIO_CFG
from vishing_ai.preprocessing.audio_preprocessing import AudioPreprocessor, PreprocessedAudio
from vishing_ai.features.acoustic import extract_acoustic_sequence
from vishing_ai.features.linguistic import (
    WhisperTranscriber,
    extract_linguistic_features,
    LinguisticResult,
)
from vishing_ai.features.behavioral import extract_behavioral_features, BehavioralResult


@dataclass
class CallFeatures:
    acoustic_seq: np.ndarray             # (max_seq_len, acoustic_dim)
    linguistic_embedding: np.ndarray     # (linguistic_embed_dim,)
    keyword_vector: np.ndarray           # (n_keywords,)
    behavioral_vector: np.ndarray        # (behavioral_dim,)
    transcript: str
    matched_keywords: list
    source_path: Optional[str] = None


class MultimodalFeatureExtractor:
    """
    High-level API: `extract(audio_path, transcript=None)` -> CallFeatures.

    If `transcript` is not supplied, Whisper is used to transcribe the audio.
    Passing a pre-existing transcript (e.g. from the synthetic dataset, which
    ships transcripts directly) avoids the cost of running ASR during
    large-scale synthetic-data experiments.
    """

    def __init__(self, audio_preprocessor: Optional[AudioPreprocessor] = None,
                 transcriber: Optional[WhisperTranscriber] = None):
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor()
        self._transcriber = transcriber  # lazy default via property

    @property
    def transcriber(self) -> WhisperTranscriber:
        if self._transcriber is None:
            self._transcriber = WhisperTranscriber()
        return self._transcriber

    def extract(self, audio_path: Union[str, Path],
                transcript: Optional[str] = None) -> Optional[CallFeatures]:
        pre: Optional[PreprocessedAudio] = self.audio_preprocessor.process(audio_path)
        if pre is None:
            return None

        if transcript is None:
            transcript = self.transcriber.transcribe(str(audio_path))

        acoustic_seq = extract_acoustic_sequence(pre)

        ling: LinguisticResult = extract_linguistic_features(transcript)
        beh: BehavioralResult = extract_behavioral_features(
            pre, transcript, hop_length=AUDIO_CFG.hop_length
        )

        return CallFeatures(
            acoustic_seq=acoustic_seq,
            linguistic_embedding=ling.token_embedding,
            keyword_vector=ling.keyword_vector,
            behavioral_vector=beh.as_vector(),
            transcript=transcript,
            matched_keywords=ling.matched_keywords,
            source_path=str(audio_path),
        )
