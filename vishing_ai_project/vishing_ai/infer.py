"""
infer.py
--------
End-to-end single-call inference: audio file -> preprocessing -> multimodal
feature extraction -> CTFN prediction -> XAI rationale, printed as JSON.

Usage:
    python -m vishing_ai.infer --audio path/to/call.wav --checkpoint checkpoints/ctfn_fold0.pt
    python -m vishing_ai.infer --audio call.wav --transcript "transcript text" --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch

from vishing_ai.models.ctfn import CTFN
from vishing_ai.features.extractor import MultimodalFeatureExtractor, CallFeatures
from vishing_ai.xai.explainer import CTFNExplainer
from vishing_ai.utils import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> CTFN:
    model = CTFN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_call(audio_path: str, checkpoint_path: str, transcript: Optional[str] = None,
                  device: Optional[torch.device] = None) -> dict:
    device = device or get_device()

    extractor = MultimodalFeatureExtractor()
    features: Optional[CallFeatures] = extractor.extract(audio_path, transcript=transcript)
    if features is None:
        raise ValueError(
            f"Audio at {audio_path} was rejected during preprocessing "
            "(likely too short). See AudioConfig.min_duration_sec."
        )

    model = load_model(checkpoint_path, device)

    batch = {
        "acoustic_seq": torch.from_numpy(features.acoustic_seq).float().unsqueeze(0).to(device),
        "linguistic_embedding": torch.from_numpy(features.linguistic_embedding).float().unsqueeze(0).to(device),
        "keyword_vector": torch.from_numpy(features.keyword_vector).float().unsqueeze(0).to(device),
        "behavioral_vector": torch.from_numpy(features.behavioral_vector).float().unsqueeze(0).to(device),
    }

    with torch.no_grad():
        output = model(**batch)

    explainer = CTFNExplainer()
    explanation = explainer.explain(output, features)

    return {
        "audio_path": str(audio_path),
        "transcript": features.transcript,
        "prediction": explanation.predicted_label,
        "confidence": explanation.confidence,
        "class_probabilities": {
            label: float(p) for label, p in zip(
                ["scam", "legitimate", "casual"], output.probs[0].tolist()
            )
        },
        "modality_contributions": explanation.modality_contributions,
        "matched_scam_keywords": explanation.top_keyword_matches,
        "behavioral_flags": explanation.behavioral_flags,
        "rationale": explanation.rationale,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=str, required=True, help="path to a call audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to a CTFN .pt checkpoint")
    parser.add_argument("--transcript", type=str, default=None,
                         help="optional pre-existing transcript (skips Whisper ASR)")
    args = parser.parse_args()

    result = predict_call(args.audio, args.checkpoint, transcript=args.transcript)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
