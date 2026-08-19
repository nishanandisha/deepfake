"""Exports the trained checkpoints into `models/` as self-contained joblib artefacts.

The raw training checkpoints are bare `state_dict`s: weights only, with no
record of the architecture that produced them. That is enough to *resume* a
run inside this repo (where the Hydra configs are on disk) but not enough to
hand someone a file they can fine-tune from, because rebuilding the module
graph requires knowing embed_dim / depth / heads / pooling / n_mfcc / ... .

Each artefact here therefore bundles three things:

  weights   -- the state_dict, on CPU so it loads without a GPU present
  config    -- the resolved model config, so the graph can be rebuilt exactly
  metadata  -- test-split metrics + provenance, so a downloaded file is
               traceable back to the run that produced it

Load one with `joblib.load(...)`, rebuild via the matching `build_*` helper,
then `load_state_dict(artefact["weights"])`. See models/README.md.

Usage:
  python scripts/export_models.py --outputs D:/deepfake-data/outputs_medium
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
from src.training.train_fusion import build_fusion_model_for_inference
from src.training.train_visual import build_visual_model

# checkpoint subpath -> (artefact name, config section, builder)
EXPORTS = {
    "visual": ("visual/checkpoints/best.pt", "visual"),
    "acoustic": ("acoustic/checkpoints/best.pt", "acoustic"),
    "fusion": ("fusion/checkpoints/best.pt", None),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build(name, cfg):
    """Rebuild the module graph so the export is verified loadable, not just copied."""
    if name == "visual":
        return build_visual_model(cfg.model.visual)
    if name == "acoustic":
        return build_acoustic_model(cfg.model.acoustic)
    return build_fusion_model_for_inference(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained models to models/ as joblib artefacts.")
    parser.add_argument("--outputs", default="D:/deepfake-data/outputs_medium")
    parser.add_argument("--data", default="medium")
    parser.add_argument("--dest", default=None, help="defaults to <repo>/models")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out = Path(args.outputs)
    dest = Path(args.dest) if args.dest else repo / "models"
    dest.mkdir(parents=True, exist_ok=True)

    config_dir = str(repo / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["model=fusion", f"data={args.data}"])

    test_results = {}
    tr_path = out / "test_results.json"
    if tr_path.exists():
        test_results = json.loads(tr_path.read_text()).get("models", {})

    policy = {}
    policy_path = out / "calibration" / "policy.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())

    manifest = []
    for name, (rel, section) in EXPORTS.items():
        ckpt = out / rel
        if not ckpt.exists():
            print(f"[skip] {name}: no checkpoint at {ckpt}")
            continue

        state = torch.load(ckpt, map_location="cpu", weights_only=False)

        # Rebuild + load so a broken export fails HERE, not for whoever
        # downloads the repo and tries to fine-tune from it.
        model = _build(name, cfg)
        model.load_state_dict(state)
        n_params = sum(p.numel() for p in model.parameters())

        # Store each config in exactly the shape its builder consumes, so a
        # loader can rebuild with `build_x(OmegaConf.create(art["config"]))`
        # and no reshaping. build_visual/acoustic_model take the section
        # itself; build_fusion_model reaches for cfg.model.* and the
        # warm-start flag, so fusion keeps the enclosing wrapper.
        if section is not None:
            model_cfg = OmegaConf.to_container(cfg.model[section], resolve=True)
        else:
            model_cfg = {
                "model": OmegaConf.to_container(cfg.model, resolve=True),
                "init_from_standalone_checkpoints": False,
            }

        artefact = {
            "name": name,
            "framework": "pytorch",
            "torch_version": torch.__version__,
            "weights": state,
            "config": model_cfg,
            "data_config": OmegaConf.to_container(cfg.data, resolve=True),
            "metadata": {
                "test_metrics": test_results.get(name if name != "fusion" else "cross_attention", {}),
                "calibration_policy": policy if name == "fusion" else {},
                "num_parameters": int(n_params),
                "source_checkpoint": str(ckpt),
                "source_sha256": _sha256(ckpt),
                "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": "LAV-DF",
            },
        }

        target = dest / f"{name}_model.joblib"
        joblib.dump(artefact, target, compress=3)
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"[ok] {name:9s} -> {target.name:26s} {size_mb:6.2f} MB  ({n_params:,} params)")
        manifest.append({
            "name": name,
            "file": target.name,
            "size_mb": round(size_mb, 2),
            "num_parameters": int(n_params),
            "sha256": _sha256(target),
            "test_auc": artefact["metadata"]["test_metrics"].get("auc"),
        })

    (dest / "manifest.json").write_text(json.dumps({
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "LAV-DF",
        "models": manifest,
    }, indent=2))
    print(f"\nManifest written to {dest / 'manifest.json'}")


if __name__ == "__main__":
    main()
