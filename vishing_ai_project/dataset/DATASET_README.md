# Synthetic Vishing Dataset

**THIS DATA IS SYNTHETICALLY GENERATED.** It contains 12000 labeled call
transcripts (4000 each for `scam`, `legitimate`, and `casual`),
produced by randomized slot-filling over hand-authored templates
(`generate_dataset.py`). It is NOT a corpus of real phone call recordings
or transcripts.

## Why synthetic
No public dataset was found that pairs real audio + transcripts with a
3-class {scam, legitimate, casual} phone-call label. A Kaggle fraud-call
dataset was evaluated and found to be text-only with no audio; its
vocabulary was used only to help seed the scam-keyword list in
`vishing_ai/features/linguistic.py`, not as training data here.

## Intended use & limitations
- Useful for validating that the end-to-end pipeline (preprocessing ->
  feature extraction -> CTFN -> XAI) runs correctly and for illustrating
  the architecture in a research paper.
- **Not suitable, on its own, for claiming real-world generalization
  performance.** Accuracy numbers obtained by training/evaluating solely on
  this dataset reflect performance on synthetic, template-structured text
  and (if paired with TTS audio) synthetic acoustic conditions. Any
  manuscript using this dataset must disclose that fact explicitly and
  should treat such numbers as an upper-bound / pipeline-validation figure,
  not a claim about real vishing-call detection accuracy.
- Templates are hand-authored and finite in number per subtype; the
  generator's `_generate_unique` retry loop guarantees exact per-class
  unique counts but does not guarantee linguistic diversity beyond the
  template + slot-filling space.
- This package ships **transcripts only**. To obtain paired audio, run
  transcripts through a TTS engine of your choice (not included) or record
  human read-throughs; the audio preprocessing pipeline
  (`vishing_ai/preprocessing/audio_preprocessing.py`) expects standard
  audio files and does not care how they were produced.

## Files
- `metadata.csv`: id, transcript, label, subtype, synthetic (always True)
