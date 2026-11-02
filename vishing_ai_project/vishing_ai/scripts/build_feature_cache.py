"""
build_feature_cache.py
-----------------------
Runs the full preprocessing + feature extraction pipeline over a dataset
directory (expects `metadata.csv` with columns id,transcript,label,...,
and optionally an `audio/` subfolder with `<id>.wav` files) and caches the
resulting `CallFeatures` to `.npz` files for fast, repeated training runs.

IMPORTANT: The bundled synthetic dataset (see `vishing_ai/data/
generate_dataset.py`) ships TRANSCRIPTS ONLY — no audio. Rows without a
matching audio file are skipped by default (`--require-audio`, on by
default) rather than silently faked, to avoid quietly training the
acoustic branch on fabricated zeros and reporting inflated results. Pass
`--allow-missing-audio` to fall back to zero-valued acoustic features for
those rows instead (NOT recommended for any reported benchmark — acoustic
features would carry zero information in that case).

Usage:
    python -m vishing_ai.scripts.build_feature_cache \\
        --dataset dataset/ --out dataset/feature_cache/
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from vishing_ai.features.extractor import MultimodalFeatureExtractor, CallFeatures
from vishing_ai.config import FEATURE_CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _zero_acoustic_seq() -> np.ndarray:
    return np.zeros((FEATURE_CFG.max_seq_len, FEATURE_CFG.acoustic_dim), dtype=np.float32)


def build_cache(dataset_dir: Path, out_dir: Path, require_audio: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_dir / "metadata.csv"
    audio_dir = dataset_dir / "audio"

    extractor = MultimodalFeatureExtractor()

    n_written, n_skipped = 0, 0
    index_rows = []

    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["id"]
            transcript = row["transcript"]
            label = row["label"]
            audio_path = audio_dir / f"{sample_id}.wav"

            if audio_path.exists():
                feats: CallFeatures | None = extractor.extract(audio_path, transcript=transcript)
                if feats is None:
                    logger.warning("Skipping %s: audio too short/corrupt", sample_id)
                    n_skipped += 1
                    continue
            elif not require_audio:
                from vishing_ai.features.linguistic import extract_linguistic_features
                from vishing_ai.features.behavioral import BehavioralResult

                ling = extract_linguistic_features(transcript)
                dummy_behavioral = BehavioralResult(0, 0, 0, 0, 0, 0).as_vector()
                feats = CallFeatures(
                    acoustic_seq=_zero_acoustic_seq(),
                    linguistic_embedding=ling.token_embedding,
                    keyword_vector=ling.keyword_vector,
                    behavioral_vector=dummy_behavioral,
                    transcript=transcript,
                    matched_keywords=ling.matched_keywords,
                    source_path=None,
                )
            else:
                n_skipped += 1
                continue

            out_path = out_dir / f"{sample_id}.npz"
            np.savez_compressed(
                out_path,
                acoustic_seq=feats.acoustic_seq,
                linguistic_embedding=feats.linguistic_embedding,
                keyword_vector=feats.keyword_vector,
                behavioral_vector=feats.behavioral_vector,
                label=label,
            )
            index_rows.append((sample_id, label, str(out_path)))
            n_written += 1

    index_path = out_dir / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "path"])
        writer.writerows(index_rows)

    logger.info("Wrote %d feature caches, skipped %d rows -> %s", n_written, n_skipped, index_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--out", type=str, default="dataset/feature_cache")
    parser.add_argument("--allow-missing-audio", action="store_true",
                         help="Fall back to zero-valued acoustic features when no audio file "
                              "is found (NOT recommended for reported benchmarks).")
    args = parser.parse_args()

    build_cache(
        Path(args.dataset), Path(args.out),
        require_audio=not args.allow_missing_audio,
    )


if __name__ == "__main__":
    main()
