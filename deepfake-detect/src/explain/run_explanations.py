"""Stage 7 driver: runs the full explanation pipeline over a split.

For each requested sample: computes the calibrated score + policy decision
(reusing Stage 6's fitted temperature/thresholds), the coarse modality
split, the fine acoustic descriptor attributions, and Grad-CAM saliency,
then writes one moderator HTML report per sample plus an aggregate
results.md carrying the attribution agreement rate.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.calibration import apply_temperature
from src.evaluation.policy import decide
from src.explain.attribution_agreement import (
    compute_attribution_agreement,
    format_agreement_markdown,
)
from src.explain.cam_visual import GradCAM, most_implicated_frames, overlay_heatmap
from src.explain.report import build_report_html, write_report
from src.explain.shap_acoustic import (
    compute_background_features,
    explain_acoustic_features,
    top_k_features,
)
from src.explain.shap_modality import compute_background_pooled, explain_modality_split
from src.models.acoustic.features import extract_acoustic_features
from src.models.fusion.late_fusion import load_frozen_checkpoint, resolve_device
from src.preprocessing.dataset import MultimodalDataset
from src.training.train_fusion import build_fusion_model
from src.utils.logging import ExperimentLogger, get_logger


def _build_dataset(cfg, split: str) -> MultimodalDataset:
    splits_dir = Path(cfg.data.splits_dir)
    return MultimodalDataset(
        splits_dir / f"{split}.csv",
        split=split,
        visual_kwargs=dict(
            frame_rate=cfg.data.frame_rate,
            frame_size=cfg.data.frame_size,
            num_frames=cfg.data.get("num_frames", 32),
        ),
        acoustic_kwargs=dict(
            sample_rate=cfg.data.audio_sample_rate,
            frame_ms=cfg.data.audio_frame_ms,
            hop_ms=cfg.data.audio_hop_ms,
            n_mfcc=cfg.model.acoustic.n_mfcc,
            pitch_tracker=cfg.data.get("pitch_tracker", "yin"),
            num_frames=cfg.data.get("num_audio_frames", 300),
        ),
        seed=cfg.seed,
    )


def _frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """[T,3,H,W] float in [0,1] -> [T,H,W,3] uint8 RGB for overlaying."""
    return (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)


def run_explanations(cfg, logger: ExperimentLogger = None) -> Dict:
    logger = logger or get_logger("explain", log_dir=Path(cfg.output_dir))
    device = resolve_device(cfg.device)

    model = build_fusion_model(cfg)
    model = load_frozen_checkpoint(model, cfg.fusion_checkpoint, device).to(device)

    policy = json.loads(Path(cfg.policy_json).read_text())
    temperature, tau_lo, tau_hi = policy["temperature"], policy["tau_lo"], policy["tau_hi"]
    logger.info(f"Loaded policy: T={temperature:.4f} tau_lo={tau_lo:.4f} tau_hi={tau_hi:.4f}")

    dataset = _build_dataset(cfg, cfg.explain_split)
    num_samples = min(cfg.max_samples, len(dataset))
    subset = Subset(dataset, list(range(num_samples)))
    loader = DataLoader(subset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)

    background_loader = DataLoader(
        Subset(dataset, list(range(min(cfg.background_samples, len(dataset))))),
        batch_size=cfg.data.batch_size, shuffle=False, num_workers=0,
    )
    background_pooled_v, background_pooled_a = compute_background_pooled(
        model, background_loader, device
    )
    background_features = compute_background_features(background_loader)

    _, feature_names = extract_acoustic_features(
        np.zeros(cfg.data.audio_sample_rate), n_mfcc=cfg.model.acoustic.n_mfcc
    )

    output_dir = Path(cfg.output_dir)
    reports_dir = output_dir / "reports"
    all_splits, all_truth_modalities, sample_records = [], [], []
    sample_index = 0

    for frames, v_mask, features, a_mask, labels in loader:
        with torch.no_grad():
            outputs = model(
                frames.to(device), v_mask.to(device), features.to(device), a_mask.to(device)
            )
        logits = outputs["y_hat_logit"].cpu().numpy()
        c_scores = apply_temperature(logits, temperature)

        modality_splits = explain_modality_split(
            model, outputs["pooled_v"], outputs["pooled_a"],
            background_pooled_v, background_pooled_a, device, n_samples=cfg.shap_samples_modality,
        )
        acoustic_attributions = explain_acoustic_features(
            model, features, a_mask, feature_names, background_features, device,
            n_samples=cfg.shap_samples_acoustic,
        )

        grad_cam = GradCAM(model.visual_encoder)
        for i in range(frames.shape[0]):
            row = dataset.df.iloc[sample_index]
            decision = decide(float(c_scores[i]), tau_lo, tau_hi)

            heatmaps = grad_cam(
                frames[i : i + 1].to(device), v_mask[i : i + 1].to(device), model.visual_aux_head
            )
            top_frames = most_implicated_frames(heatmaps, top_k=cfg.top_k_frames)
            frames_uint8 = _frames_to_uint8(frames[i])
            overlaid = [overlay_heatmap(frames_uint8[idx], heatmaps[idx]) for idx, _ in top_frames]

            sample_id = str(row.get("sample_id", f"sample_{sample_index}"))
            html = build_report_html(
                sample_id=sample_id,
                authenticity_score=float(c_scores[i]),
                decision=decision,
                modality_split=modality_splits[i],
                top_acoustic_features=top_k_features(
                    acoustic_attributions[i], k=cfg.top_k_features
                ),
                saliency_frames=overlaid,
                frame_indices=[idx for idx, _ in top_frames],
                ground_truth_label=row.get("label"),
                ground_truth_modality=row.get("manipulated_modality"),
            )
            safe_name = sample_id.replace("/", "_").replace("\\", "_")
            write_report(str(reports_dir / f"{safe_name}.html"), html)

            all_splits.append(modality_splits[i])
            all_truth_modalities.append(str(row.get("manipulated_modality", "none")))
            sample_records.append(
                {
                    "sample_id": sample_id,
                    "authenticity_score": float(c_scores[i]),
                    "decision": decision,
                    "visual_share": modality_splits[i]["visual_share"],
                    "label": row.get("label"),
                    "manipulated_modality": row.get("manipulated_modality"),
                }
            )
            sample_index += 1

        logger.info(f"Explained {sample_index}/{num_samples} samples")

    agreement = compute_attribution_agreement(
        all_splits, all_truth_modalities, dominance_margin=cfg.dominance_margin
    )
    logger.info(
        f"Attribution agreement rate: {agreement['attribution_agreement_rate']} "
        f"({agreement['num_correct']}/{agreement['num_considered']})"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "explanations.json").write_text(
        json.dumps(
            {"samples": sample_records,
             "attribution_agreement_rate": agreement["attribution_agreement_rate"],
             "num_considered": agreement["num_considered"]},
            indent=2,
        )
    )

    results_md = "\n".join(
        [
            "# Explanation layer -- results",
            "",
            f"Split: `{cfg.explain_split}` &middot; samples explained: {sample_index}",
            f"&middot; reports: `{reports_dir}`",
            "",
            format_agreement_markdown(agreement),
        ]
    )
    (output_dir / "results.md").write_text(results_md + "\n")

    return agreement


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Run via scripts/explain.py (Hydra entry point).")
