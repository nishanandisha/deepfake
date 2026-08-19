"""End-to-end sanity check for Stage 7: trains a tiny fusion model, fits
the Stage 6 policy, then runs the full explanation pipeline and verifies
the moderator reports, explanations.json, and attribution agreement rate
all come out. Mechanics-only, like the other stages' e2e tests.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from src.evaluation.calibrate_and_select_policy import run_calibration_and_policy
from src.explain.run_explanations import run_explanations
from src.training.train_fusion import train_fusion
from src.utils.seed import set_seed


def _write_video(path: Path, value: int, num_frames=4, size=(48, 48)) -> bool:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, size)
    if not writer.isOpened():
        return False
    for _ in range(num_frames):
        writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    writer.release()
    return True


def _write_tone(path: Path, freq: float, amplitude: float, duration_s=0.3, sample_rate=16000):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), sample_rate)


def _build_split_csv(tmp_path: Path, name: str, n_real: int, n_fake: int) -> Path:
    rows = []
    for i in range(n_real):
        video_path = tmp_path / f"{name}_real_{i}.mp4"
        audio_path = tmp_path / f"{name}_real_{i}.wav"
        if not _write_video(video_path, value=10):
            pytest.skip("mp4v codec unavailable in this environment")
        _write_tone(audio_path, freq=110, amplitude=0.02)
        rows.append(
            {"sample_id": f"{name}_real_{i}", "video_path": str(video_path),
             "audio_path": str(audio_path), "label": "real",
             "manipulated_modality": "none", "identity_id": f"{name}_r{i}"}
        )
    for i in range(n_fake):
        video_path = tmp_path / f"{name}_fake_{i}.mp4"
        audio_path = tmp_path / f"{name}_fake_{i}.wav"
        if not _write_video(video_path, value=240):
            pytest.skip("mp4v codec unavailable in this environment")
        _write_tone(audio_path, freq=4000, amplitude=0.8)
        # Alternate the ground-truth manipulated modality so the agreement
        # computation sees more than one class to score against.
        modality = ["video", "audio", "both"][i % 3]
        rows.append(
            {"sample_id": f"{name}_fake_{i}", "video_path": str(video_path),
             "audio_path": str(audio_path), "label": "fake",
             "manipulated_modality": modality, "identity_id": f"{name}_f{i}"}
        )

    df = pd.DataFrame(rows)
    csv_path = tmp_path / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _base_cfg(splits_dir: Path):
    return {
        "data": {
            "splits_dir": str(splits_dir),
            "frame_rate": 8, "frame_size": 48, "num_frames": 4,
            "audio_sample_rate": 16000, "audio_frame_ms": 25, "audio_hop_ms": 10,
            "num_audio_frames": 40, "batch_size": 4,
            "use_class_balanced_sampler": False, "num_workers": 0,
        },
        "model": {
            "visual": {
                "backbone": "efficientnet_b0", "pretrained": False, "embed_dim": 32,
                "transformer": {"depth": 1, "heads": 4, "ff_dim": 32, "dropout": 0.0},
            },
            "acoustic": {
                "n_mfcc": 13, "embed_dim": 32,
                "transformer": {"depth": 1, "heads": 4, "ff_dim": 32, "dropout": 0.0},
            },
            "cross_attention": {"heads": 4, "dropout": 0.0},
            "gate": {"hidden_dim": 16},
            "aux_loss": {"lambda_visual": 0.3, "lambda_acoustic": 0.3},
        },
    }


@pytest.mark.slow
def test_explanation_pipeline_end_to_end(tmp_path: Path):
    set_seed(0, deterministic=False)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _build_split_csv(splits_dir, "train", n_real=6, n_fake=6)
    _build_split_csv(splits_dir, "val", n_real=4, n_fake=4)
    _build_split_csv(splits_dir, "calibration", n_real=4, n_fake=4)
    _build_split_csv(splits_dir, "test", n_real=3, n_fake=3)

    base = _base_cfg(splits_dir)
    outputs = tmp_path / "outputs"

    train_fusion(
        OmegaConf.create(
            {
                "seed": 0, "device": "auto", "output_dir": str(outputs / "fusion"),
                "init_from_standalone_checkpoints": False,
                "late_fusion_results": str(outputs / "late_fusion" / "results.json"),
                **base,
                "training": {
                    "lr": 1e-3, "weight_decay": 0.0, "warmup_epochs": 1, "max_epochs": 4,
                    "grad_clip_norm": 1.0,
                    "checkpoint_dir": str(outputs / "fusion" / "checkpoints"),
                    "early_stopping_patience": 4,
                },
            }
        )
    )

    run_calibration_and_policy(
        OmegaConf.create(
            {
                "seed": 0, "device": "auto", "output_dir": str(outputs / "calibration"),
                "fusion_checkpoint": str(outputs / "fusion" / "checkpoints" / "best.pt"),
                "false_suppression_ceiling": 0.02, "threshold_grid_size": 50,
                "tau_hi": None, "tau_lo": None, **base,
            }
        )
    )

    agreement = run_explanations(
        OmegaConf.create(
            {
                "seed": 0, "device": "auto", "output_dir": str(outputs / "explain"),
                "fusion_checkpoint": str(outputs / "fusion" / "checkpoints" / "best.pt"),
                "policy_json": str(outputs / "calibration" / "policy.json"),
                "explain_split": "test",
                "max_samples": 4, "background_samples": 4,
                "shap_samples_modality": 16, "shap_samples_acoustic": 32,
                "top_k_features": 5, "top_k_frames": 2, "dominance_margin": 0.15,
                **base,
            }
        )
    )

    explain_dir = outputs / "explain"
    assert (explain_dir / "results.md").exists()
    assert (explain_dir / "explanations.json").exists()

    reports = list((explain_dir / "reports").glob("*.html"))
    assert len(reports) == 4, f"expected one report per explained sample, got {len(reports)}"
    report_text = reports[0].read_text(encoding="utf-8")
    assert "Modality split" in report_text
    assert "data:image/png;base64," in report_text  # Grad-CAM frames embedded

    records = json.loads((explain_dir / "explanations.json").read_text())
    assert len(records["samples"]) == 4
    for sample in records["samples"]:
        assert sample["decision"] in {"approve", "flag", "block"}
        assert 0.0 <= sample["visual_share"] <= 1.0

    # The test split has 3 fakes with labeled modalities among the 4
    # explained samples, so agreement must be computed over a non-empty set.
    assert agreement["num_considered"] > 0
    assert 0.0 <= agreement["attribution_agreement_rate"] <= 1.0
