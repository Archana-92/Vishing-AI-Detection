"""
linguistic.py
-------------
Transcribes call audio with OpenAI Whisper and derives linguistic features:
  1. A bag-of-tokens embedding of the transcript (hashed vocabulary — see
     the Limitations section of README.md for why this is a disclosed
     simplification rather than a learned contextual embedding).
  2. A scam-keyword match vector: counts / presence of terms drawn from a
     curated vocabulary associated with vishing scripts (urgency, payment
     requests, authority impersonation, secrecy coercion, etc).

The keyword vocabulary can be extended from external sources; the project
notes (see project memory) used a text-only Kaggle fraud-call dataset
*solely* to broaden this vocabulary — never as training audio, since it
contains no audio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

import numpy as np

from vishing_ai.config import FEATURE_CFG

# ---------------------------------------------------------------------
# Scam keyword vocabulary (curated + extended from external text corpus)
# Grouped by rhetorical function for interpretability in the XAI layer.
# ---------------------------------------------------------------------
SCAM_KEYWORD_GROUPS = {
    "urgency": [
        "immediately", "right now", "urgent", "act now", "expire", "expires",
        "final notice", "last chance", "within 24 hours", "today only",
    ],
    "authority_impersonation": [
        "irs", "social security", "police department", "federal agent",
        "warrant", "court order", "arrest", "government agency", "tax office",
    ],
    "payment_coercion": [
        "gift card", "wire transfer", "bitcoin", "crypto", "bank account number",
        "routing number", "prepaid card", "western union", "pay now", "processing fee",
    ],
    "secrecy_isolation": [
        "don't tell anyone", "keep this confidential", "don't hang up",
        "stay on the line", "don't call anyone else", "this is confidential",
    ],
    "credential_phishing": [
        "verify your account", "confirm your password", "one time code",
        "otp", "security code", "pin number", "social security number", "ssn",
    ],
}

_ALL_KEYWORDS: List[str] = [kw for group in SCAM_KEYWORD_GROUPS.values() for kw in group]


@dataclass
class LinguisticResult:
    transcript: str
    token_embedding: np.ndarray          # (linguistic_embed_dim,)
    keyword_vector: np.ndarray           # (len(_ALL_KEYWORDS),) binary/count
    matched_keywords: List[str] = field(default_factory=list)


class WhisperTranscriber:
    """Thin wrapper around openai-whisper for call-audio transcription."""

    def __init__(self, model_size: str = FEATURE_CFG.whisper_model_size):
        self.model_size = model_size
        self._model = None  # lazy-loaded (Whisper checkpoints are large)

    @property
    def model(self):
        if self._model is None:
            import whisper  # local import: heavy dependency, only load if used
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path, fp16=False, language="en")
        return result.get("text", "").strip()


def _hash_token(token: str, vocab_size: int) -> int:
    return (hash(token) % vocab_size + vocab_size) % vocab_size


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


@lru_cache(maxsize=1)
def _keyword_pattern():
    escaped = sorted((re.escape(k) for k in _ALL_KEYWORDS), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


def bag_of_tokens_embedding(text: str, dim: int = FEATURE_CFG.linguistic_embed_dim,
                             vocab_size: int = FEATURE_CFG.vocab_size) -> np.ndarray:
    """
    Hashed bag-of-tokens embedding.

    DISCLOSED LIMITATION: this is a simple, non-contextual representation
    (hashing trick + random projection), not a learned contextual sentence
    embedding. It is fast and dependency-light, which suits a reproducible
    reference pipeline, but it cannot capture word order or semantics
    beyond lexical overlap. See README limitations section. Swapping in a
    sentence-transformer embedding is a natural extension for future work.
    """
    tokens = _tokenize(text)
    if not tokens:
        return np.zeros(dim, dtype=np.float32)

    counts = np.zeros(vocab_size, dtype=np.float32)
    for tok in tokens:
        counts[_hash_token(tok, vocab_size)] += 1.0
    counts /= (np.linalg.norm(counts) + 1e-8)

    # deterministic random projection vocab_size -> dim (seeded, reproducible)
    rng = np.random.RandomState(1234)
    projection = rng.normal(scale=1.0 / np.sqrt(vocab_size), size=(vocab_size, dim))
    embedding = counts @ projection
    return embedding.astype(np.float32)


def extract_keyword_vector(text: str) -> tuple[np.ndarray, List[str]]:
    text_lower = text.lower()
    vec = np.zeros(len(_ALL_KEYWORDS), dtype=np.float32)
    matched = []
    for i, kw in enumerate(_ALL_KEYWORDS):
        if kw in text_lower:
            vec[i] = 1.0
            matched.append(kw)
    return vec, matched


def extract_linguistic_features(transcript: str) -> LinguisticResult:
    embedding = bag_of_tokens_embedding(transcript)
    keyword_vec, matched = extract_keyword_vector(transcript)
    return LinguisticResult(
        transcript=transcript,
        token_embedding=embedding,
        keyword_vector=keyword_vec,
        matched_keywords=matched,
    )
