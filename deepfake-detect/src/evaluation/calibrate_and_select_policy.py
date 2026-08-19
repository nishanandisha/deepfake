"""Stage 6 entry point: fits temperature scaling on the calibration split,
selects approve/flag/block thresholds on the validation split, and writes
a results markdown with the reliability diagram and operating-point table.
"""

import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.calibration import (
    apply_temperature,
    calibrated_fake_probability,
    compute_reliability_diagram,
    fit_temperature,
    get_fusion_logits,
    plot_reliability_diagram,
)
from src.evaluation.policy import resolve_thresholds
from src.models.fusion.late_fusion import load_frozen_checkpoint, resolve_device
from src.preprocessing.dataset import MultimodalDataset
from src.training.train_fusion import build_fusion_model
from src.utils.logging import ExperimentLogger, get_logger


def _build_dataset(cfg, split: str) -> MultimodalDataset:
    splits_dir = Path(cfg.data.splits_dir)
    visual_kwargs = dict(
        frame_rate=cfg.data.frame_rate,
        frame_size=cfg.data.frame_size,
        num_frames=cfg.data.get("num_frames", 32),
    )
    acoustic_kwargs = dict(
        sample_rate=cfg.data.audio_sample_rate,
        frame_ms=cfg.data.audio_frame_ms,
        hop_ms=cfg.data.audio_hop_ms,
        n_mfcc=cfg.model.acoustic.n_mfcc,
        pitch_tracker=cfg.data.get("pitch_tracker", "yin"),
        num_frames=cfg.data.get("num_audio_frames", 300),
    )
    return MultimodalDataset(
        splits_dir / f"{split}.csv", split=split,
        visual_kwargs=visual_kwargs, acoustic_kwargs=acoustic_kwargs, seed=cfg.seed,
    )


def run_calibration_and_policy(cfg, logger: ExperimentLogger = None) -> Dict:
    logger = logger or get_logger("calibration", log_dir=Path(cfg.output_dir))
    device = resolve_device(cfg.device)

    model = build_fusion_model(cfg)
    model = load_frozen_checkpoint(model, cfg.fusion_checkpoint, device).to(device)

    calibration_dataset = _build_dataset(cfg, "calibration")
    val_dataset = _build_dataset(cfg, "val")

    logger.info(f"Computing fusion logits on {len(calibration_dataset)} calibration samples")
    calib_logits, calib_labels = get_fusion_logits(
        model, calibration_dataset, device, cfg.data.batch_size, cfg.data.num_workers
    )
    logger.info(f"Computing fusion logits on {len(val_dataset)} val samples")
    val_logits, val_labels = get_fusion_logits(
        model, val_dataset, device, cfg.data.batch_size, cfg.data.num_workers
    )

    temperature = fit_temperature(calib_logits, calib_labels)
    logger.info(f"Fitted temperature T={temperature:.4f} on calibration split")

    # Reliability shown on val (out-of-sample from the T fit) so it's a
    # genuine check that calibration generalizes, not just fits the
    # calibration split it was optimized on.
    val_probs_before = calibrated_fake_probability(val_logits)
    val_probs_after = calibrated_fake_probability(val_logits, temperature)
    reliability_before = compute_reliability_diagram(val_probs_before, val_labels)
    reliability_after = compute_reliability_diagram(val_probs_after, val_labels)
    logger.info(f"ECE before={reliability_before['ece']:.4f} after={reliability_after['ece']:.4f}")

    diagram_path = str(Path(cfg.output_dir) / "reliability_diagram.png")
    plot_reliability_diagram(reliability_before, reliability_after, diagram_path)

    val_c_scores = apply_temperature(val_logits, temperature)
    thresholds = resolve_thresholds(
        val_c_scores, val_labels,
        tau_hi=cfg.get("tau_hi"), tau_lo=cfg.get("tau_lo"),
        false_suppression_ceiling=cfg.false_suppression_ceiling,
        grid_size=cfg.threshold_grid_size,
    )
    logger.info(f"Selected thresholds: {thresholds}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {"temperature": temperature, **thresholds}
    (output_dir / "policy.json").write_text(json.dumps(artifact, indent=2))

    _write_results_markdown(
        output_dir / "results.md", temperature, thresholds,
        reliability_before, reliability_after, "reliability_diagram.png",
        cfg.false_suppression_ceiling,
    )

    return artifact


def _write_results_markdown(
    output_path: Path,
    temperature: float,
    thresholds: Dict,
    reliability_before: Dict,
    reliability_after: Dict,
    diagram_filename: str,
    false_suppression_ceiling: float,
) -> None:
    ceiling_ok = thresholds["false_suppression_rate"] <= false_suppression_ceiling
    ece_improved = reliability_after["ece"] < reliability_before["ece"]
    lines = [
        "# Calibration & decision policy -- results",
        "",
        f"**Temperature (fit on calibration split):** T = {temperature:.4f}",
        "",
        f"**Calibration quality (val split, out-of-sample):** "
        f"ECE before = {reliability_before['ece']:.4f}, "
        f"ECE after = {reliability_after['ece']:.4f} "
        f"({'improved' if ece_improved else 'NOT improved'})",
        "",
        f"![reliability diagram]({diagram_filename})",
        "",
        "## Operating point (selected on val split)",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| tau_lo (block below) | {thresholds['tau_lo']:.4f} |",
        f"| tau_hi (approve at/above) | {thresholds['tau_hi']:.4f} |",
        f"| False-suppression rate | {thresholds['false_suppression_rate']:.4f} |",
        f"| False-suppression ceiling | {false_suppression_ceiling:.4f} "
        f"({'OK' if ceiling_ok else 'VIOLATED'}) |",
        f"| Review-queue rate | {thresholds['review_queue_rate']:.4f} |",
        f"| Detection recall | {thresholds['detection_recall']:.4f} |",
    ]
    output_path.write_text("\n".join(lines) + "\n")
