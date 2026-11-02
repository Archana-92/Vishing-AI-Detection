"""
vishing_ai
==========

A multimodal deep learning system for detecting voice phishing (vishing) calls.

This package implements the Convolutional Transformer Fusion Network (CTFN),
a novel architecture that fuses acoustic, linguistic, and behavioral signals
extracted from phone call audio to classify calls as `scam`, `legitimate`,
or `casual`, together with a lightweight explainability (XAI) layer that
produces human-readable rationales for each prediction.

Author: Research implementation for academic publication.

IMPORTANT RESEARCH DISCLOSURE
------------------------------
The reference dataset shipped with this package (see `data/generate_dataset.py`)
is SYNTHETICALLY GENERATED from randomized slot-filling templates. It is not
a corpus of real phone call recordings. Any accuracy figures obtained by
training on this dataset alone characterize performance on synthetic,
template-structured data and should NOT be reported as real-world
generalization performance. See `README.md` -> "Limitations & Honesty
Statement" for full detail. This disclosure is intentionally duplicated in
multiple places in the codebase so it cannot be easily overlooked when
figures are copied into a manuscript.
"""

__version__ = "1.0.0"
__all__ = ["preprocessing", "features", "models", "xai"]
