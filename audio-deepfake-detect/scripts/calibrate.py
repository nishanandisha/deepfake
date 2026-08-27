"""Fits an AUDIO-ONLY decision policy for the acoustic model.

Two steps, on two different splits, neither of them test:
  1. temperature T on the CALIBRATION split (minimising NLL)
  2. approve/flag/block thresholds on the VALIDATION split, subject to a
     hard ceiling on the false-suppression rate (authentic audio wrongly
     blocked)

Writes outputs/calibration/policy.json, which scripts/predict.py and
src/inference/pipeline.py consume via --policy.

The shipped artefact deliberately carries no policy: the parent multimodal
project fitted one on the *fused* audio-visual logit, and those thresholds
do not transfer to this model's differently-scaled logit. Fit your own here
against your own calibration split.

Usage:
  python scripts/calibrate.py --config-name calibration
"""

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration import (
    apply_temperature,
    calibrated_fake_probability,
    compute_reliability_diagram,
    fit_temperature,
    get_acoustic_logits,
    plot_reliability_diagram,
)
from src.evaluation.datasets import build_split_dataset
from src.evaluation.policy import resolve_thresholds
from src.inference.loader import load_acoustic_model
from src.training.common import resolve_device
from src.utils.logging import get_logger
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="calibration")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    output_dir = Path(cfg.output_dir)
    logger = get_logger("calibration", log_dir=output_dir)
    device = resolve_device(cfg.device)

    loaded = load_acoustic_model(cfg.model_artefact, device)

    calibration_dataset = build_split_dataset(cfg, "calibration", loaded.n_mfcc)
    val_dataset = build_split_dataset(cfg, "val", loaded.n_mfcc)

    logger.info(f"Fitting temperature on {len(calibration_dataset)} calibration samples")
    calib_logits, calib_labels = get_acoustic_logits(
        loaded.model, calibration_dataset, device,
        batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
    )
    temperature = fit_temperature(calib_logits, calib_labels)
    logger.info(f"T = {temperature:.4f}")

    before = compute_reliability_diagram(calibrated_fake_probability(calib_logits), calib_labels)
    after = compute_reliability_diagram(
        calibrated_fake_probability(calib_logits, temperature), calib_labels
    )
    plot_reliability_diagram(before, after, str(output_dir / "reliability.png"))
    logger.info(f"ECE {before['ece']:.4f} -> {after['ece']:.4f}")

    logger.info(f"Selecting thresholds on {len(val_dataset)} validation samples")
    val_logits, val_labels = get_acoustic_logits(
        loaded.model, val_dataset, device,
        batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
    )
    policy = resolve_thresholds(
        apply_temperature(val_logits, temperature), val_labels,
        tau_hi=cfg.get("tau_hi"), tau_lo=cfg.get("tau_lo"),
        false_suppression_ceiling=cfg.false_suppression_ceiling,
        grid_size=cfg.threshold_grid_size,
    )
    policy["temperature"] = temperature
    policy["ece_before"], policy["ece_after"] = before["ece"], after["ece"]
    policy["modality"] = "audio"

    policy_path = output_dir / "policy.json"
    policy_path.write_text(json.dumps(policy, indent=2))
    logger.info(f"Wrote {policy_path}:\n{json.dumps(policy, indent=2)}")


if __name__ == "__main__":
    main()
