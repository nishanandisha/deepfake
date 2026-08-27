"""Identity-disjoint train/val/calibration/test splitting for FakeAVCeleb.

No identity_id may appear in more than one split -- entire identities are
assigned to a split as a unit, then a greedy balancing heuristic picks which
split each identity goes to next so that, in aggregate, every split ends up
with a similar real:fake ratio and a similar mix of the four
(real/real, fake-video/real-audio, real-video/fake-audio, fake/fake)
manipulation pairings. Exact per-identity stratification is impossible
(a single identity may contribute samples to several pairings at once), so
this is a best-effort balance, not a guarantee every split matches the
global ratio exactly -- check split_stats() output after splitting.

DFDC, when present, must never be passed to this module -- it stays fully
held out for cross-dataset evaluation.
"""

import random
from typing import Dict, List

import pandas as pd

PAIRING_COLUMNS = [
    "none",  # real video, real audio
    "video",  # fake video, real audio
    "audio",  # real video, fake audio
    "both",  # fake video, fake audio
]

DEFAULT_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "calibration": 0.05, "test": 0.10}


def _identity_profiles(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per identity_id with counts per manipulation pairing and
    overall real/fake counts, used to drive the greedy split assignment."""
    profile = (
        manifest.groupby(["identity_id", "manipulated_modality"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=PAIRING_COLUMNS, fill_value=0)
    )
    profile["real_count"] = profile["none"]
    profile["fake_count"] = profile[["video", "audio", "both"]].sum(axis=1)
    profile["total"] = profile["real_count"] + profile["fake_count"]
    return profile


def make_identity_disjoint_splits(
    manifest: pd.DataFrame,
    split_ratios: Dict[str, float] = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Assign every identity_id in `manifest` wholesale to exactly one split,
    greedily balancing real/fake ratio and the four manipulation pairings.

    Returns {split_name: manifest_subset_dataframe}.
    """
    split_ratios = split_ratios or DEFAULT_SPLIT_RATIOS
    assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, "split_ratios must sum to 1.0"

    profiles = _identity_profiles(manifest)
    identities = list(profiles.index)
    random.Random(seed).shuffle(identities)

    balance_columns = PAIRING_COLUMNS + ["real_count", "fake_count"]
    totals = profiles[balance_columns].sum()

    targets = {split: totals * ratio for split, ratio in split_ratios.items()}
    running = {split: pd.Series(0, index=balance_columns, dtype=float) for split in split_ratios}
    assignment: Dict[str, List[str]] = {split: [] for split in split_ratios}

    for identity_id in identities:
        row = profiles.loc[identity_id, balance_columns]

        # Assign to whichever split is currently furthest *under* its target
        # share (as a fraction of that split's total target), summed across
        # the balance columns -- a simple greedy multi-label group balancer.
        def deficit(split: str) -> float:
            target = targets[split].replace(0, 1e-9)
            return float(((target - running[split]) / target).sum())

        best_split = max(split_ratios, key=deficit)
        assignment[best_split].append(identity_id)
        running[best_split] = running[best_split] + row

    identity_to_split = {
        identity_id: split for split, ids in assignment.items() for identity_id in ids
    }
    manifest = manifest.copy()
    manifest["split"] = manifest["identity_id"].map(identity_to_split)

    return {
        split: manifest[manifest["split"] == split].drop(columns=["split"])
        for split in split_ratios
    }


def assert_no_identity_leakage(splits: Dict[str, pd.DataFrame]) -> None:
    """Fail loudly if any identity_id appears in more than one split."""
    seen: Dict[str, str] = {}
    for split_name, df in splits.items():
        for identity_id in df["identity_id"].unique():
            if identity_id in seen and seen[identity_id] != split_name:
                raise AssertionError(
                    f"Identity leakage detected: identity_id={identity_id!r} appears in "
                    f"both split {seen[identity_id]!r} and {split_name!r}."
                )
            seen[identity_id] = split_name


def split_stats(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-split sample counts, identity counts, real/fake ratio, and the
    four-way manipulation-pairing breakdown."""
    rows = []
    for split_name, df in splits.items():
        row = {
            "split": split_name,
            "num_samples": len(df),
            "num_identities": df["identity_id"].nunique(),
            "num_real": int((df["label"] == "real").sum()),
            "num_fake": int((df["label"] == "fake").sum()),
        }
        pairing_counts = df["manipulated_modality"].value_counts()
        for pairing in PAIRING_COLUMNS:
            row[f"pairing_{pairing}"] = int(pairing_counts.get(pairing, 0))
        rows.append(row)
    return pd.DataFrame(rows)
