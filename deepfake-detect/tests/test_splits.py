import pandas as pd
import pytest

from src.preprocessing.splits import (
    assert_no_identity_leakage,
    make_identity_disjoint_splits,
    split_stats,
)


def _synthetic_manifest(num_identities: int = 40) -> pd.DataFrame:
    rows = []
    pairings = ["none", "video", "audio", "both"]
    for i in range(num_identities):
        identity_id = f"id{i:03d}"
        # Every identity contributes a few real samples plus one manipulation
        # pairing, roughly mimicking FakeAVCeleb's per-identity structure.
        for j in range(3):
            rows.append(
                {
                    "sample_id": f"{identity_id}_real_{j}",
                    "video_path": f"/fake/{identity_id}_real_{j}.mp4",
                    "audio_path": f"/fake/{identity_id}_real_{j}.mp4",
                    "identity_id": identity_id,
                    "label": "real",
                    "manipulated_modality": "none",
                    "source_generator": None,
                }
            )
        pairing = pairings[i % 3 + 1]
        for j in range(5):
            rows.append(
                {
                    "sample_id": f"{identity_id}_fake_{j}",
                    "video_path": f"/fake/{identity_id}_fake_{j}.mp4",
                    "audio_path": f"/fake/{identity_id}_fake_{j}.mp4",
                    "identity_id": identity_id,
                    "label": "fake",
                    "manipulated_modality": pairing,
                    "source_generator": "some_method",
                }
            )
    return pd.DataFrame(rows)


def test_splits_are_identity_disjoint():
    manifest = _synthetic_manifest()
    splits = make_identity_disjoint_splits(manifest, seed=0)
    assert_no_identity_leakage(splits)  # must not raise


def test_identity_leakage_is_caught():
    manifest = _synthetic_manifest(num_identities=10)
    splits = make_identity_disjoint_splits(manifest, seed=0)

    # Deliberately inject a leak: copy one row from train into val.
    leaked_identity = splits["train"]["identity_id"].iloc[0]
    leaked_row = splits["train"][splits["train"]["identity_id"] == leaked_identity].iloc[[0]]
    splits["val"] = pd.concat([splits["val"], leaked_row], ignore_index=True)

    with pytest.raises(AssertionError, match="Identity leakage"):
        assert_no_identity_leakage(splits)


def test_split_ratios_are_roughly_respected():
    manifest = _synthetic_manifest(num_identities=100)
    ratios = {"train": 0.7, "val": 0.15, "calibration": 0.05, "test": 0.10}
    splits = make_identity_disjoint_splits(manifest, split_ratios=ratios, seed=1)
    total = sum(len(df) for df in splits.values())

    assert abs(len(splits["train"]) / total - 0.70) < 0.10
    assert abs(len(splits["test"]) / total - 0.10) < 0.08


def test_no_split_is_empty_with_enough_identities():
    manifest = _synthetic_manifest(num_identities=100)
    splits = make_identity_disjoint_splits(manifest, seed=2)
    for split_name, df in splits.items():
        assert len(df) > 0, f"split {split_name} is empty"


def test_split_stats_reports_expected_columns():
    manifest = _synthetic_manifest(num_identities=40)
    splits = make_identity_disjoint_splits(manifest, seed=3)
    stats = split_stats(splits)

    assert set(stats["split"]) == set(splits.keys())
    assert "num_real" in stats.columns
    assert "pairing_both" in stats.columns
