"""Split the manifest without leaking a speaker, a recording, or a generator.

Two independent constraints, both of which the naive random split violates:

1. **Group disjointness.** Every clip is a chunk of a longer recording. All
   chunks of `yt_0000` must live in exactly one split, or the model gets
   graded on recordings it already memorised.

2. **Source holdout.** A detector that has heard ElevenLabs is not thereby
   shown to detect *synthetic speech*; it may only detect ElevenLabs. So a
   subset of generators is withheld from training entirely and appears only
   in the test split. The EER restricted to those rows is the honest
   generalisation number, and it is the one worth reporting.

The real half of this dataset comes from just 14 YouTube videos, so there is
no meaningful speaker diversity to preserve beyond the group constraint --
a limitation of the corpus that `scripts/build_splits.py` prints loudly
rather than hides.
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# Withheld from training so the test split contains generators the model has
# never heard. Kokoro and Hume are the two smallest fake sources (68 + 116
# clips), which keeps the training set as large as possible while still
# buying a genuine unseen-generator measurement.
DEFAULT_HELD_OUT_SOURCES = ("hume", "kokoro")

DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "calibration": 0.05, "test": 0.10}


def _allocate_groups(
    group_sizes: pd.Series, ratios: Dict[str, float], seed: int
) -> Dict[str, str]:
    """Assign whole groups to splits, greedily matching target clip counts.

    Groups are wildly uneven here (yt_0000 alone is 104 clips against a
    median of ~15), so proportional-by-count beats proportional-by-group:
    handing out 10% of the *groups* could hand out 40% of the *clips*.
    Largest-first placement keeps the biggest group from blowing past a
    small split's entire budget.
    """
    rng = np.random.default_rng(seed)
    total = float(group_sizes.sum())
    targets = {split: ratio * total for split, ratio in ratios.items()}
    current = {split: 0.0 for split in ratios}
    assignment: Dict[str, str] = {}

    # Shuffle first so ties among equal-sized groups are not resolved by name,
    # then place largest-first.
    order = list(group_sizes.index)
    rng.shuffle(order)
    order.sort(key=lambda g: group_sizes[g], reverse=True)

    for group in order:
        size = float(group_sizes[group])
        # Whichever split is furthest below its target, in absolute clips.
        deficits = {s: targets[s] - current[s] for s in ratios}
        chosen = max(deficits, key=deficits.get)
        assignment[group] = chosen
        current[chosen] += size

    return assignment


def make_splits(
    manifest: pd.DataFrame,
    held_out_sources: Sequence[str] = DEFAULT_HELD_OUT_SOURCES,
    ratios: Dict[str, float] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Returns the manifest with a `split` column added.

    Held-out sources are forced entirely into `test`. Everything else is
    allocated group-wise, per label, so the class balance of each split
    tracks the corpus rather than drifting with the group lottery.
    """
    ratios = dict(ratios or DEFAULT_RATIOS)
    held_out = set(held_out_sources)

    unknown = held_out - set(manifest["source"].unique())
    if unknown:
        raise ValueError(f"held_out_sources not present in manifest: {sorted(unknown)}")

    out = manifest.copy()
    out["split"] = pd.NA

    # 1. Generators withheld from training go straight to test.
    out.loc[out["source"].isin(held_out), "split"] = "test"

    # 2. Everything else is allocated by group, independently per label so
    #    real and AI rows are both represented in every split.
    remaining = out["split"].isna()
    for label in sorted(out.loc[remaining, "label"].unique()):
        mask = remaining & (out["label"] == label)
        sizes = out.loc[mask].groupby("group").size()
        assignment = _allocate_groups(sizes, ratios, seed)
        out.loc[mask, "split"] = out.loc[mask, "group"].map(assignment)

    if out["split"].isna().any():
        raise RuntimeError("internal error: some rows were never assigned a split")

    return out


def assert_no_group_leakage(splits: pd.DataFrame) -> None:
    """Every group must belong to exactly one split."""
    offenders = (
        splits.groupby("group")["split"].nunique().loc[lambda s: s > 1].index.tolist()
    )
    if offenders:
        raise AssertionError(
            f"{len(offenders)} group(s) span multiple splits: {offenders[:10]}"
        )


def assert_sources_held_out(
    splits: pd.DataFrame, held_out_sources: Sequence[str]
) -> None:
    """Held-out generators must appear in test and nowhere else."""
    for source in held_out_sources:
        elsewhere = splits[(splits["source"] == source) & (splits["split"] != "test")]
        if len(elsewhere):
            raise AssertionError(
                f"held-out source {source!r} leaked into "
                f"{sorted(elsewhere['split'].unique())}"
            )


def split_stats(splits: pd.DataFrame) -> pd.DataFrame:
    """Clip / group / class-balance table per split."""
    rows: List[dict] = []
    for split, frame in splits.groupby("split"):
        rows.append(
            {
                "split": split,
                "clips": len(frame),
                "groups": frame["group"].nunique(),
                "human": int((frame["label"] == 0).sum()),
                "ai": int((frame["label"] == 1).sum()),
                "sources": ",".join(sorted(frame["source"].unique())),
            }
        )
    order = {"train": 0, "val": 1, "calibration": 2, "test": 3}
    return pd.DataFrame(rows).sort_values("split", key=lambda c: c.map(order))
