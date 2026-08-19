import pandas as pd

from src.preprocessing.subset import subset_manifest, subset_stats, warn_if_too_few_real


def _manifest(n_real: int, n_fake_per_pairing: int) -> pd.DataFrame:
    rows = []
    for i in range(n_real):
        rows.append(
            {"sample_id": f"real_{i}", "video_path": f"/r{i}.mp4", "audio_path": f"/r{i}.mp4",
             "identity_id": f"id_r{i}", "label": "real",
             "manipulated_modality": "none", "source_generator": None}
        )
    for pairing in ["video", "audio", "both"]:
        for i in range(n_fake_per_pairing):
            rows.append(
                {"sample_id": f"{pairing}_{i}", "video_path": f"/{pairing}{i}.mp4",
                 "audio_path": f"/{pairing}{i}.mp4", "identity_id": f"id_{pairing}{i}",
                 "label": "fake", "manipulated_modality": pairing, "source_generator": "gen"}
            )
    return pd.DataFrame(rows)


def test_subset_hits_target_size():
    manifest = _manifest(n_real=500, n_fake_per_pairing=3000)
    subset = subset_manifest(manifest, target_size=2000, seed=0)
    assert len(subset) == 2000


def test_scarce_real_class_is_fully_preserved():
    """The core reason this module exists: a naive random subset of a
    40:1-skewed dataset leaves single-digit real clips per split."""
    manifest = _manifest(n_real=500, n_fake_per_pairing=6500)  # ~40:1 skew
    subset = subset_manifest(manifest, target_size=2000, seed=0, keep_all_real=True)

    stats = subset_stats(subset)
    assert stats["num_real"] == 500  # every real clip kept
    assert stats["real_fraction"] == 0.25
    # A naive 2000/20000 random sample would have yielded ~50 real clips.
    assert stats["num_real"] > 50


def test_balanced_dataset_does_not_collapse_to_one_class():
    """Regression: LAV-DF is ~27% real, and `keep_all_real` used to take
    every real clip that fit in the budget. On a balanced dataset that
    filled all 2,000 slots with real clips and left zero fakes -- a subset
    with one class is untrainable and its AUC is undefined."""
    manifest = _manifest(n_real=3900, n_fake_per_pairing=3600)  # ~27% real, like LAV-DF
    subset = subset_manifest(manifest, target_size=2000, seed=0, keep_all_real=True)

    stats = subset_stats(subset)
    assert stats["num_fake"] > 0, "subset must contain fakes"
    assert stats["num_real"] > 0, "subset must contain real clips"
    assert stats["real_fraction"] == 0.25
    assert stats["num_real"] == 500 and stats["num_fake"] == 1500


def test_all_manipulation_pairings_survive_subsetting():
    """Stage 7's attribution agreement needs video/audio/both all present."""
    manifest = _manifest(n_real=500, n_fake_per_pairing=3000)
    subset = subset_manifest(manifest, target_size=1000, seed=0)

    pairings = subset[subset["label"] == "fake"]["manipulated_modality"].unique()
    assert set(pairings) == {"video", "audio", "both"}


def test_subset_larger_than_dataset_returns_everything():
    manifest = _manifest(n_real=10, n_fake_per_pairing=10)
    subset = subset_manifest(manifest, target_size=10_000, seed=0)
    assert len(subset) == len(manifest)


def test_keep_all_real_false_respects_target_fraction():
    manifest = _manifest(n_real=500, n_fake_per_pairing=3000)
    subset = subset_manifest(
        manifest, target_size=1000, target_real_fraction=0.1, seed=0, keep_all_real=False
    )
    stats = subset_stats(subset)
    assert stats["num_real"] == 100


def test_warn_when_too_few_real_clips():
    manifest = _manifest(n_real=20, n_fake_per_pairing=1000)
    subset = subset_manifest(manifest, target_size=500, seed=0)
    stats = subset_stats(subset)

    warning = warn_if_too_few_real(stats, min_real=200)
    assert warning is not None
    assert "20 real clips" in warning


def test_no_warning_when_enough_real_clips():
    manifest = _manifest(n_real=500, n_fake_per_pairing=3000)
    subset = subset_manifest(manifest, target_size=2000, seed=0)
    assert warn_if_too_few_real(subset_stats(subset), min_real=200) is None


def test_subset_is_deterministic_for_a_seed():
    manifest = _manifest(n_real=200, n_fake_per_pairing=1000)
    first = subset_manifest(manifest, target_size=500, seed=7)
    second = subset_manifest(manifest, target_size=500, seed=7)
    assert list(first["sample_id"]) == list(second["sample_id"])
