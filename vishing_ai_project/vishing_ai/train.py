"""
train.py
--------
Phase 3 (training) of the pipeline: stratified k-fold cross-validation
training of the CTFN model over a cached feature set produced by
`vishing_ai/scripts/build_feature_cache.py`.

Reports per-fold and mean +/- std accuracy and macro-F1 to
`reports/cv_results.json`, and saves the best-epoch checkpoint of each fold
to `checkpoints/`.

HONESTY NOTE: results produced by this script when run against the bundled
synthetic dataset characterize performance on synthetic, template-generated
transcripts (and, unless paired with real/TTS audio, zero-valued acoustic
features — see build_feature_cache.py). Do not report such numbers as
general vishing-detection accuracy without stating the data source. This
script does not claim, target, or validate any specific accuracy figure;
it reports whatever the model actually achieves on the data it is given.

Usage:
    python -m vishing_ai.train --cache dataset/feature_cache --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from vishing_ai.config import TRAIN_CFG, CHECKPOINT_DIR, REPORTS_DIR, LABELS, NUM_CLASSES
from vishing_ai.models.ctfn import CTFN
from vishing_ai.utils import (
    VishingDataset, LabeledCallFeatures, collate_batch,
    set_seed, get_device, accuracy, macro_f1,
)
from vishing_ai.features.extractor import CallFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_cached_dataset(cache_dir: Path) -> VishingDataset:
    index_path = cache_dir / "index.csv"
    items: List[LabeledCallFeatures] = []
    with open(index_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = np.load(row["path"])
            feats = CallFeatures(
                acoustic_seq=data["acoustic_seq"],
                linguistic_embedding=data["linguistic_embedding"],
                keyword_vector=data["keyword_vector"],
                behavioral_vector=data["behavioral_vector"],
                transcript="",
                matched_keywords=[],
            )
            items.append(LabeledCallFeatures(features=feats, label=row["label"]))
    logger.info("Loaded %d cached samples from %s", len(items), index_path)
    return VishingDataset(items)


def stratified_kfold_indices(labels: List[str], k: int, seed: int) -> List[tuple]:
    """Simple stratified k-fold split without external dependencies."""
    rng = np.random.RandomState(seed)
    label_to_indices = {}
    for i, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(i)
    for lbl in label_to_indices:
        rng.shuffle(label_to_indices[lbl])

    folds = [[] for _ in range(k)]
    for lbl, idxs in label_to_indices.items():
        for j, idx in enumerate(idxs):
            folds[j % k].append(idx)

    splits = []
    all_idx = set(range(len(labels)))
    for i in range(k):
        val_idx = folds[i]
        train_idx = list(all_idx - set(val_idx))
        splits.append((train_idx, val_idx))
    return splits


def run_epoch(model, loader, device, optimizer=None, criterion=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    all_logits, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        acoustic = batch["acoustic_seq"].to(device)
        linguistic = batch["linguistic_embedding"].to(device)
        keywords = batch["keyword_vector"].to(device)
        behavioral = batch["behavioral_vector"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(is_train):
            output = model(acoustic, linguistic, keywords, behavioral)
            loss = criterion(output.logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG.grad_clip_norm)
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_logits.append(output.logits.detach())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy(all_logits, all_labels),
        "macro_f1": macro_f1(all_logits, all_labels, NUM_CLASSES),
    }


def train_fold(fold_idx: int, dataset: VishingDataset, train_idx: List[int], val_idx: List[int],
               device: torch.device, epochs: int) -> dict:
    set_seed(TRAIN_CFG.seed + fold_idx)

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=TRAIN_CFG.batch_size,
        shuffle=True, collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=TRAIN_CFG.batch_size,
        shuffle=False, collate_fn=collate_batch,
    )

    model = CTFN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG.learning_rate,
                                   weight_decay=TRAIN_CFG.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=TRAIN_CFG.label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer, criterion)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None, criterion=criterion)
        scheduler.step()

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        logger.info(
            "[fold %d][epoch %02d] train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f val_f1=%.3f",
            fold_idx, epoch, train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"], val_metrics["macro_f1"],
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CFG.early_stopping_patience:
                logger.info("[fold %d] Early stopping at epoch %d", fold_idx, epoch)
                break

    checkpoint_path = CHECKPOINT_DIR / f"ctfn_fold{fold_idx}.pt"
    torch.save(best_state, checkpoint_path)

    model.load_state_dict(best_state)
    final_val_metrics = run_epoch(model, val_loader, device, optimizer=None, criterion=criterion)

    return {
        "fold": fold_idx,
        "best_val_macro_f1": best_val_f1,
        "final_val_metrics": final_val_metrics,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=str, default="dataset/feature_cache")
    parser.add_argument("--epochs", type=int, default=TRAIN_CFG.epochs)
    parser.add_argument("--k-folds", type=int, default=TRAIN_CFG.k_folds)
    args = parser.parse_args()

    set_seed(TRAIN_CFG.seed)
    device = get_device(TRAIN_CFG.device)
    logger.info("Using device: %s", device)

    dataset = load_cached_dataset(Path(args.cache))
    labels = [item.label for item in dataset.items]

    splits = stratified_kfold_indices(labels, k=args.k_folds, seed=TRAIN_CFG.seed)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        result = train_fold(fold_idx, dataset, train_idx, val_idx, device, epochs=args.epochs)
        fold_results.append(result)

    accs = [r["final_val_metrics"]["accuracy"] for r in fold_results]
    f1s = [r["final_val_metrics"]["macro_f1"] for r in fold_results]

    summary = {
        "labels": LABELS,
        "k_folds": args.k_folds,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "per_fold": [
            {"fold": r["fold"], "accuracy": r["final_val_metrics"]["accuracy"],
             "macro_f1": r["final_val_metrics"]["macro_f1"]}
            for r in fold_results
        ],
        "disclosure": (
            "Reported metrics reflect performance on the dataset located at "
            f"'{args.cache}'. If this is the bundled synthetic dataset, these "
            "numbers characterize synthetic-data performance ONLY and must not "
            "be reported as real-world vishing detection accuracy without "
            "explicit disclosure of the synthetic data source."
        ),
    }

    report_path = REPORTS_DIR / "cv_results.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Mean accuracy: %.4f +/- %.4f", summary["mean_accuracy"], summary["std_accuracy"])
    logger.info("Mean macro-F1: %.4f +/- %.4f", summary["mean_macro_f1"], summary["std_macro_f1"])
    logger.info("Full report written to %s", report_path)


if __name__ == "__main__":
    main()
