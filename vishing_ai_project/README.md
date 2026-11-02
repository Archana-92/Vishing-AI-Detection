# vishing_ai — Multimodal Voice Phishing Detection

A research implementation of a **Convolutional Transformer Fusion Network
(CTFN)**: a multimodal deep learning architecture that fuses acoustic,
linguistic, and behavioral signals extracted from phone call audio to
classify calls as `scam`, `legitimate`, or `casual`, together with an
attention-based explainability (XAI) layer that produces a human-readable
rationale for every prediction.

This repository is structured as an installable Python package
(`vishing_ai`) with four pipeline phases, a synthetic reference dataset,
training/inference scripts, and a smoke-test suite — all of which have been
executed in this environment (not just written) to confirm they run
end-to-end. See **Verification** below for exactly what was run.

---

## 1. Architecture

```
                 ┌───────────────────────┐
   raw audio ──► │ Audio Preprocessing   │  Librosa: MFCC, STFT magnitude,
                 │ (Phase 1)             │  F0 contour (pYIN), RMS energy
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │ Multimodal Feature     │  acoustic seq | linguistic embed +
                 │ Extraction (Phase 2)   │  scam-keyword vector | behavioral stats
                 └──────────┬────────────┘
                            │
     ┌──────────────────────┼──────────────────────┐
     ▼                      ▼                      ▼
┌──────────┐         ┌─────────────┐         ┌─────────────┐
│ 1D-CNN   │         │ Linguistic  │         │ Behavioral  │
│    +     │         │ MLP branch  │         │ MLP branch  │
│Transformer│        └─────┬───────┘         └──────┬──────┘
│ encoder  │               │                        │
└────┬─────┘               │                        │
     └──────────► Cross-Modal Attention Fusion ◄─────┘
                  (shared embedding space, Phase 3 model = CTFN)
                            │
                  ┌─────────▼─────────┐
                  │ Classification MLP │──► scam / legitimate / casual
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │ XAI rationale      │  (Phase 4) attention weights +
                  │ builder            │  signal-level checks → text
                  └────────────────────┘
```

**Acoustic branch**: a 1D-CNN stack (`AcousticEncoder` in
`models/ctfn.py`) extracts local temporal patterns (phoneme-scale
transients in pitch/energy), followed by a multi-head Transformer encoder
for long-range prosodic dependencies across the call, with learned
attention pooling.

**Linguistic branch**: a hashed bag-of-tokens embedding of the Whisper
transcript, concatenated with a binary scam-keyword match vector across
five rhetorical categories (urgency, authority impersonation, payment
coercion, secrecy/isolation, credential phishing), passed through an MLP.

**Behavioral branch**: a 6-dimensional statistical summary (speech rate,
pause count, hesitation index, energy variance, pitch jitter, talk/silence
ratio) passed through a small MLP.

**Fusion**: `CrossModalAttentionFusion` projects all three branch
embeddings into a shared space and combines them via learned attention
(not naive concatenation), exposing per-modality attention weights that
the XAI layer reports as "modality contribution."

**Activations**: all nonlinearities use `TrainableSwish` — Swish with a
per-channel learnable β parameter — rather than fixed ReLU/GELU.

---

## 2. Repository layout

```
vishing_ai/
  config.py                  central hyperparameters & paths
  preprocessing/
    audio_preprocessing.py   Phase 1: Librosa-based signal extraction
  features/
    acoustic.py               frame-level acoustic feature sequence
    linguistic.py              Whisper ASR + bag-of-tokens + scam keywords
    behavioral.py               speech-rate / pause / hesitation statistics
    extractor.py                 orchestrates the three branches (Phase 2)
  models/
    layers.py                  TrainableSwish, positional encoding, fusion layer
    ctfn.py                     the CTFN model (Phase 3)
  xai/
    explainer.py                attention → human-readable rationale (Phase 4)
  data/
    generate_dataset.py         synthetic dataset generator
  scripts/
    build_feature_cache.py      caches extracted features to .npz for training
  train.py                      stratified k-fold cross-validation training
  infer.py                      single-call end-to-end inference + XAI
  tests/
    test_smoke.py                dataset, feature-shape, and forward-pass tests
setup.py
requirements.txt
```

---

## 3. Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python ≥3.10. Whisper model weights are downloaded on first use
of the linguistic branch's ASR path (not required if you supply
transcripts directly, as the synthetic dataset does).

---

## 4. Usage

```bash
# 1. Generate the synthetic reference dataset (12,000 transcripts)
python -m vishing_ai.data.generate_dataset --out dataset/ --n-per-class 4000 --zip

# 2. Build a feature cache (requires dataset/audio/<id>.wav per row for the
#    acoustic branch to carry real signal — see Limitations below)
python -m vishing_ai.scripts.build_feature_cache --dataset dataset/ --out dataset/feature_cache

# 3. Train with stratified k-fold cross-validation
python -m vishing_ai.train --cache dataset/feature_cache --epochs 40 --k-folds 5

# 4. Run inference on a single call with an XAI rationale
python -m vishing_ai.infer --audio call.wav --checkpoint checkpoints/ctfn_fold0.pt
```

---

## 5. Verification performed in this environment

Rather than only writing code, the following was actually executed here:

| Step | Result |
|---|---|
| `py_compile` on every module | all files compile cleanly |
| `pytest vishing_ai/tests/test_smoke.py` | **3/3 passed** — dataset balance/uniqueness, linguistic feature shapes, full CTFN forward pass with valid softmax + attention outputs |
| Full-scale synthetic dataset generation (`--n-per-class 4000`) | **12,000/12,000 unique transcripts**, exactly 4,000 per class, `synthetic=True` on every row |
| `build_feature_cache.py` on a 60-row slice | feature caching succeeded, 60/60 `.npz` files written |
| `train.py` on the cached slice (2 folds × 2 epochs) | training loop ran end-to-end: loss decreased, checkpoints saved, `reports/cv_results.json` written with the honesty disclosure field |
| `infer.py` on a synthetic test waveform + explicit transcript | full pipeline (preprocessing → feature extraction → CTFN → XAI) ran without error and returned a well-formed prediction, modality attention weights, matched keywords, and rationale text |

Two real bugs were caught and fixed during this process (not just written
correctly on the first try): a channel-broadcasting bug in `TrainableSwish`
between the conv branch (`B, C, T`) and linear branches (`B, C`), and
insufficient template entropy in the synthetic-dataset generator that
initially failed to reach 4,000 unique samples per class.

**Not executed here**: a full 40-epoch, 5-fold training run on real audio,
and live Whisper transcription (the sandboxed network in this environment
cannot reach Whisper's model-weight host). Nothing in this repository
reports or implies a specific accuracy figure as an achieved result —
`train.py` only reports whatever the model actually gets on the data you
give it, plus the disclosure below.

---

## 6. Limitations & Honesty Statement

This section is intentionally explicit and duplicated in code comments
(`vishing_ai/__init__.py`, `data/generate_dataset.py`, `train.py`) so it
cannot be easily dropped when results are copied into a manuscript.

1. **No accuracy target is assumed or reported.** No 97–98%-style
   benchmark number appears anywhere in this codebase as an achieved or
   implied result. Any number your own training run produces should be
   reported alongside the exact dataset and configuration used to obtain
   it.
2. **The bundled dataset is synthetic.** `synthetic_vishing_dataset.zip` /
   `dataset/metadata.csv` contains 12,000 call **transcripts** generated
   from randomized slot-filling over hand-authored templates — not real
   call recordings or transcripts. Every row is tagged `synthetic=True`.
   A manuscript using this data must state that explicitly; results from
   it characterize pipeline correctness and synthetic-data performance,
   not real-world vishing-detection accuracy.
3. **The dataset ships transcripts only, no audio.** To train the acoustic
   branch on real signal, pair these transcripts with TTS-generated or
   human-recorded audio yourself; `build_feature_cache.py` will otherwise
   either skip rows lacking audio (default) or fall back to zero-valued
   acoustic features (`--allow-missing-audio`, explicitly discouraged for
   reported benchmarks since it removes acoustic information entirely).
4. **A real-world Kaggle fraud-call dataset was evaluated and found
   unsuitable** as a primary training source because its rows are
   text-only with no paired audio. It was used only to help broaden the
   scam-keyword vocabulary in `features/linguistic.py`, never as training
   data.
5. **The linguistic embedding is a disclosed simplification.** It is a
   hashed bag-of-tokens representation with a fixed random projection —
   not a learned contextual sentence embedding (e.g. a transformer sentence
   encoder). It cannot capture word order or deep semantics beyond lexical
   overlap; this is a stated direction for future work, not a hidden
   limitation.
6. **The scam-keyword vocabulary is curated and finite** (five rhetorical
   categories, see `SCAM_KEYWORD_GROUPS`). It will miss scam scripts using
   different phrasing and should be reported as a coverage limitation, not
   a general-purpose lexicon.
7. **XAI is attention-based, not causal.** `xai/explainer.py` reports
   attention weights and concrete signal checks (matched keywords,
   behavioral flags) as a rationale. Attention mass is a widely used but
   imperfect proxy for feature importance; it is not a substitute for a
   causal attribution method (e.g. integrated gradients, SHAP) if a
   reviewer requires a more rigorous faithfulness guarantee.
8. **Behavioral-flag thresholds are descriptive heuristics**
   (`CTFNExplainer.thresholds`), not statistically validated cutoffs
   derived from held-out data — stated explicitly in the code.

---

## 7. Citing the synthetic dataset

If you use `synthetic_vishing_dataset.zip` in a publication, cite it as:
*template-generated synthetic transcripts, N=12,000, produced by
`vishing_ai/data/generate_dataset.py`, not real call recordings.*
