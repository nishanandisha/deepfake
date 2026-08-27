"""Unpacks models/fusion_model.joblib into the checkpoint + policy.json layout
that scripts/serve.py expects.

The exported joblib artefact bundles weights, config and the fitted decision
policy in one file, but serve.py takes the two paths the *training* run wrote
(a bare state_dict and calibration/policy.json) -- and that outputs/ tree only
existed on the original training machine. This bridges the two so the repo can
be served from a fresh clone with no training data present.

Usage:
  python scripts/setup_local_checkpoint.py
"""

import json
import sys
from pathlib import Path

import joblib
import torch

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    artefact_path = REPO / "models" / "fusion_model.joblib"
    if not artefact_path.exists():
        sys.exit(f"missing {artefact_path}")

    artefact = joblib.load(artefact_path)

    checkpoint = REPO / "outputs" / "exported" / "fusion" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artefact["weights"], checkpoint)

    policy = REPO / "outputs" / "exported" / "calibration" / "policy.json"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(json.dumps(artefact["metadata"]["calibration_policy"], indent=2))

    print(f"checkpoint -> {checkpoint.relative_to(REPO)}")
    print(f"policy     -> {policy.relative_to(REPO)}")


if __name__ == "__main__":
    main()
