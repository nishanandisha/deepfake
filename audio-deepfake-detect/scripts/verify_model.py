"""Verifies models/acoustic_model.joblib is genuinely usable.

Reads ONLY the joblib file (never a training checkpoint or the Hydra
configs), rebuilds the graph from the embedded config, loads the weights
strictly, and runs a real forward pass. That distinction matters -- an
artefact that merely copied bytes would still `joblib.load` fine and look
healthy. This is what catches a config/weights mismatch.

Exit code is non-zero on failure, so it can gate a commit.

Usage:
  python scripts/verify_model.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.loader import DEFAULT_MODEL_PATH, audio_settings_from, load_acoustic_model


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH)
    if not path.exists():
        print(f"FAIL: no artefact at {path}")
        return 1

    try:
        loaded = load_acoustic_model(str(path))
        settings = audio_settings_from(loaded.data_config)

        features = torch.randn(2, settings["num_audio_frames"], loaded.input_dim)
        mask = torch.zeros(2, settings["num_audio_frames"], dtype=torch.bool)
        mask[1, -50:] = True  # exercise the padding path too

        with torch.no_grad():
            logits = loaded.model(features, padding_mask=mask)

        assert logits.shape == (2,), f"expected logits [2], got {tuple(logits.shape)}"
        assert torch.isfinite(logits).all(), "non-finite output"

        n_params = sum(p.numel() for p in loaded.model.parameters())
        auc = loaded.test_metrics.get("auc")
        auc_s = f"{auc:.4f}" if isinstance(auc, (int, float)) else "n/a"

        print(f"  [PASS] {path.name}  {path.stat().st_size / 1048576:.2f} MB  "
              f"{n_params:,} params  test AUC {auc_s}  out={tuple(logits.shape)}")
        print("\nRebuilt from the embedded config, weights loaded strictly, "
              "forward pass produced finite output.")
        return 0
    except Exception as exc:  # noqa: BLE001 -- report the failure, don't traceback-dump
        print(f"  [FAIL] {path.name}  {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
