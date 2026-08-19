"""Verifies every artefact in `models/` is genuinely usable for fine-tuning.

Deliberately independent of export_models.py: it reads ONLY the joblib file
(never the original checkpoint or the Hydra configs), rebuilds the graph from
the embedded config, loads the weights strictly, and runs a real forward pass.

That distinction matters -- an export that merely copied bytes would still
`joblib.load` fine and look healthy. This is what catches a config/weights
mismatch that would only otherwise surface for whoever clones the repo.

Exit code is non-zero if any artefact fails, so it can gate a commit.

Usage:
  python scripts/verify_models.py
"""

import sys
from pathlib import Path

import joblib
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_acoustic import build_acoustic_model
from src.training.train_fusion import build_fusion_model_for_inference
from src.training.train_visual import build_visual_model


def _rebuild(name, cfg):
    if name == "visual":
        return build_visual_model(cfg)
    if name == "acoustic":
        return build_acoustic_model(cfg)
    return build_fusion_model_for_inference(cfg)


def _forward(name, model, data_cfg):
    """One real forward pass with correctly-shaped dummy input."""
    b = 1
    n_frames = data_cfg.get("num_frames", 16)
    size = data_cfg.get("frame_size", 112)
    n_audio = data_cfg.get("num_audio_frames", 400)

    frames = torch.rand(b, n_frames, 3, size, size)
    v_mask = torch.zeros(b, n_frames, dtype=torch.bool)

    if name == "visual":
        return model(frames, v_mask)

    # Acoustic feature width is whatever that branch's input norm expects.
    # The standalone model exposes it as `encoder`, the fusion model nests
    # it as `acoustic_encoder`.
    enc = model.encoder if name == "acoustic" else model.acoustic_encoder
    feats = torch.randn(b, n_audio, enc.input_norm.num_features)
    a_mask = torch.zeros(b, n_audio, dtype=torch.bool)

    if name == "acoustic":
        return model(feats, a_mask)
    return model(frames, v_mask, feats, a_mask)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    models_dir = repo / "models"

    if not models_dir.is_dir():
        print(f"FAIL: models/ directory does not exist at {models_dir}")
        return 1

    artefacts = sorted(models_dir.glob("*.joblib"))
    if not artefacts:
        print(f"FAIL: no .joblib artefacts found in {models_dir}")
        return 1

    print(f"Verifying {len(artefacts)} artefact(s) in {models_dir}\n")
    failures = []

    for path in artefacts:
        name = path.stem.replace("_model", "")
        try:
            art = joblib.load(path)
            cfg = OmegaConf.create(art["config"])
            model = _rebuild(name, cfg)
            model.load_state_dict(art["weights"], strict=True)
            model.eval()

            with torch.no_grad():
                out = _forward(name, model, art.get("data_config", {}))

            logit = out["y_hat_logit"] if isinstance(out, dict) else out
            n_params = sum(p.numel() for p in model.parameters())
            auc = art["metadata"].get("test_metrics", {}).get("auc")
            auc_s = f"{auc:.4f}" if isinstance(auc, (int, float)) else "n/a"

            assert torch.isfinite(logit).all(), "non-finite output"

            print(f"  [PASS] {path.name:26s} {path.stat().st_size/1048576:6.2f} MB  "
                  f"{n_params:>9,} params  test AUC {auc_s}  out={tuple(logit.shape)}")
        except Exception as exc:  # noqa: BLE001 -- report every failure, don't stop at the first
            print(f"  [FAIL] {path.name:26s} {type(exc).__name__}: {exc}")
            failures.append(path.name)

    if failures:
        print(f"\n{len(failures)} artefact(s) FAILED verification: {', '.join(failures)}")
        return 1

    print(f"\nAll {len(artefacts)} artefact(s) verified: rebuilt from embedded config, "
          f"weights loaded strictly, forward pass produced finite output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
