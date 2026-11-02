"""
test_smoke.py
-------------
Lightweight smoke tests that don't require GPU or downloaded model weights:
1. Synthetic dataset generation produces exact, unique, class-balanced counts.
2. Linguistic feature extraction (bag-of-tokens + keyword vector) is shape-correct.
3. Behavioral feature extraction produces a fixed-length vector.
4. CTFN forward pass runs end-to-end on random tensors of the correct shape
   and produces valid probability distributions and attention weights.

Run with: pytest vishing_ai/tests/test_smoke.py -v
"""

import numpy as np
import torch

from vishing_ai.config import FEATURE_CFG, NUM_CLASSES
from vishing_ai.data.generate_dataset import generate_dataset
from vishing_ai.features.linguistic import extract_linguistic_features, _ALL_KEYWORDS
from vishing_ai.models.ctfn import CTFN, NUM_KEYWORDS


def test_synthetic_dataset_balanced_and_unique():
    rows = generate_dataset(n_per_class=50)
    assert len(rows) == 150
    labels = [r[1] for r in rows]
    assert labels.count("scam") == 50
    assert labels.count("legitimate") == 50
    assert labels.count("casual") == 50
    transcripts = [r[0] for r in rows]
    assert len(set(transcripts)) == len(transcripts), "transcripts must be unique"
    assert all(r[3] is True for r in rows), "synthetic flag must always be True"


def test_linguistic_features_shape():
    result = extract_linguistic_features(
        "This is the IRS calling, you must pay immediately with a gift card."
    )
    assert result.token_embedding.shape == (FEATURE_CFG.linguistic_embed_dim,)
    assert result.keyword_vector.shape == (len(_ALL_KEYWORDS),)
    assert "immediately" in result.matched_keywords
    assert "gift card" in result.matched_keywords


def test_ctfn_forward_pass():
    torch.manual_seed(0)
    model = CTFN()
    model.eval()

    batch_size = 4
    acoustic = torch.randn(batch_size, FEATURE_CFG.max_seq_len, FEATURE_CFG.acoustic_dim)
    linguistic = torch.randn(batch_size, FEATURE_CFG.linguistic_embed_dim)
    keywords = torch.zeros(batch_size, NUM_KEYWORDS)
    behavioral = torch.randn(batch_size, FEATURE_CFG.behavioral_dim)

    with torch.no_grad():
        output = model(acoustic, linguistic, keywords, behavioral)

    assert output.logits.shape == (batch_size, NUM_CLASSES)
    probs_sum = output.probs.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones(batch_size), atol=1e-4)

    assert output.modality_weights.shape == (batch_size, 3)
    mod_sum = output.modality_weights.sum(dim=-1)
    assert torch.allclose(mod_sum, torch.ones(batch_size), atol=1e-4)

    assert output.temporal_attention.shape == (batch_size, FEATURE_CFG.max_seq_len)


if __name__ == "__main__":
    test_synthetic_dataset_balanced_and_unique()
    test_linguistic_features_shape()
    test_ctfn_forward_pass()
    print("All smoke tests passed.")
