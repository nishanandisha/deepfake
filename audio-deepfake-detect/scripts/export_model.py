"""Exports a trained acoustic checkpoint into models/ as a self-contained
joblib artefact.

A raw training checkpoint is a bare `state_dict`: weights only, with no
record of the architecture that produced them. That is enough to *resume* a
run inside this repo (where the Hydra configs are on disk) but not enough to
hand someone a file they can load elsewhere, because rebuilding the module
graph requires knowing embed_dim / depth / heads / pooling / n_mfcc.

The artefact therefore bundles:

  weights   -- the state_dict, on CPU so it loads without a GPU present
  config    -- the resolved model config, so the graph can be rebuilt exactly
  metadata  -- test metrics + provenance, so the file is traceable back to
               the run that produced it

Usage:
  python scripts/export_model.py --checkpoint outputs/acoustic_branch/checkpoints/best.pt
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_acoustic import build_acoustic_model


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the acoustic model to models/.")
    parser.add_argument("--checkpoint", default="outputs/acoustic_branch/checkpoints/best.pt")
    parser.add_argument("--data", default="default", help="which configs/data preset to record")
    parser.add_argument("--test-results", default=None,
                        help="results.json from scripts/evaluate.py, embedded as test metrics")
    parser.add_argument("--dest", default=None, help="defaults to <repo>/models")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"No checkpoint at {checkpoint}")

    dest = Path(args.dest) if args.dest else repo / "models"
    dest.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        cfg = compose(config_name="config", overrides=[f"data={args.data}"])

    test_metrics = {}
    if args.test_results and Path(args.test_results).exists():
        test_metrics = json.loads(Path(args.test_results).read_text())

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Rebuild + load so a broken export fails HERE, not for whoever picks
    # the file up later.
    model = build_acoustic_model(cfg.model)
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())

    artefact = {
        "name": "acoustic",
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "weights": state,
        # Stored in exactly the shape build_acoustic_model consumes, so a
        # loader can rebuild with no reshaping.
        "config": OmegaConf.to_container(cfg.model, resolve=True),
        "data_config": OmegaConf.to_container(cfg.data, resolve=True),
        "metadata": {
            "test_metrics": test_metrics,
            "calibration_policy": {},
            "num_parameters": int(n_params),
            "source_checkpoint": str(checkpoint),
            "source_sha256": _sha256(checkpoint),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": cfg.data.get("name", "unknown"),
        },
    }

    target = dest / "acoustic_model.joblib"
    joblib.dump(artefact, target, compress=3)
    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"[ok] acoustic -> {target.name}  {size_mb:.2f} MB  ({n_params:,} params)")

    (dest / "manifest.json").write_text(json.dumps({
        "exported_at": artefact["metadata"]["exported_at"],
        "dataset": artefact["metadata"]["dataset"],
        "modality": "audio",
        "models": [{
            "name": "acoustic",
            "file": target.name,
            "size_mb": round(size_mb, 2),
            "num_parameters": int(n_params),
            "sha256": _sha256(target),
            "test_auc": test_metrics.get("auc"),
        }],
    }, indent=2))
    print(f"Manifest written to {dest / 'manifest.json'}")


if __name__ == "__main__":
    main()
