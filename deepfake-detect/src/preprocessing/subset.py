"""Class-aware subsetting for the lean training configuration.

Naively sampling N clips from a deepfake dataset is a trap: these datasets
run heavily fake-skewed (FakeAVCeleb is ~40:1), so a random 2,000-clip
subset yields only ~50 real clips. After identity-disjoint splitting across
four splits that leaves single-digit real samples per split -- too few for a
stable AUC, and it makes the false-suppression rate (computed *only* over
real samples, and a hard constraint in Stage 6) essentially undefined.

So: keep every real clip, then sample fakes down to hit a target ratio,
stratifying across the manipulation pairings so video-only/audio-only/both
all survive into the subset (Stage 7's attribution agreement needs all
three to be present).
"""

from typing import Dict, Optional

import pandas as pd

PAIRINGS = ["video", "audio", "both"]


def subset_manifest(
    manifest: pd.DataFrame,
    target_size: int,
    target_real_fraction: float = 0.25,
    seed: int = 42,
    keep_all_real: bool = True,
) -> pd.DataFrame:
    """Returns a class-aware subset of `manifest`.

    target_real_fraction is a goal, not a guarantee -- if the dataset simply
    doesn't contain enough real clips, every available real clip is kept and
    the achieved ratio will be lower (check subset_stats afterwards).
    Fakes are sampled evenly across the three manipulation pairings so no
    pairing is dropped entirely.
    """
    real = manifest[manifest["label"] == "real"]
    fake = manifest[manifest["label"] == "fake"]

    target_real = int(round(target_size * target_real_fraction))

    # Take the target number of real clips, or every real clip there is if
    # the dataset has fewer than that.
    #
    # An earlier version took `min(len(real), target_size)` when
    # keep_all_real was set, reasoning that real clips are always the scarce
    # class. That holds for FakeAVCeleb (~2.5% real) but NOT for a balanced
    # dataset like LAV-DF (~27% real), where it silently consumed the entire
    # budget with real clips and produced a subset containing zero fakes --
    # untrainable, and AUC undefined. Never let one class fill the budget.
    n_real = min(target_real, len(real))
    if keep_all_real and len(real) < target_real:
        n_real = len(real)  # scarce-real case: don't sample down further

    real_subset = real.sample(n=n_real, random_state=seed) if n_real < len(real) else real

    n_fake = max(target_size - len(real_subset), 0)
    fake_subset = _stratified_fake_sample(fake, n_fake, seed)

    combined = pd.concat([real_subset, fake_subset], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _stratified_fake_sample(fake: pd.DataFrame, n_fake: int, seed: int) -> pd.DataFrame:
    """Samples `n_fake` fakes spread as evenly as possible across the three
    manipulation pairings, redistributing any shortfall from
    under-populated pairings to the others."""
    if n_fake <= 0 or fake.empty:
        return fake.iloc[0:0]
    if n_fake >= len(fake):
        return fake

    groups = {p: fake[fake["manipulated_modality"] == p] for p in PAIRINGS}
    groups = {p: g for p, g in groups.items() if not g.empty}

    # Any fake rows with an unexpected pairing value still deserve a shot at
    # inclusion rather than being silently dropped.
    other = fake[~fake["manipulated_modality"].isin(PAIRINGS)]
    if not other.empty:
        groups["__other__"] = other

    if not groups:
        return fake.sample(n=n_fake, random_state=seed)

    quota = {p: n_fake // len(groups) for p in groups}
    remainder = n_fake - sum(quota.values())
    for pairing in list(groups)[:remainder]:
        quota[pairing] += 1

    # Redistribute quota from pairings that can't fill theirs.
    shortfall = 0
    for pairing, group in groups.items():
        if quota[pairing] > len(group):
            shortfall += quota[pairing] - len(group)
            quota[pairing] = len(group)

    while shortfall > 0:
        grew = False
        for pairing, group in groups.items():
            if shortfall <= 0:
                break
            headroom = len(group) - quota[pairing]
            if headroom > 0:
                take = min(headroom, shortfall)
                quota[pairing] += take
                shortfall -= take
                grew = True
        if not grew:  # every pairing exhausted
            break

    sampled = [
        group.sample(n=quota[pairing], random_state=seed)
        for pairing, group in groups.items()
        if quota[pairing] > 0
    ]
    return pd.concat(sampled, ignore_index=True) if sampled else fake.iloc[0:0]


def subset_stats(subset: pd.DataFrame) -> Dict[str, object]:
    """Sanity figures to print after subsetting -- verify the real fraction
    is high enough for AUC and false-suppression rate to be meaningful."""
    total = len(subset)
    num_real = int((subset["label"] == "real").sum())
    pairing_counts = subset["manipulated_modality"].value_counts().to_dict()

    return {
        "num_samples": total,
        "num_real": num_real,
        "num_fake": total - num_real,
        "real_fraction": num_real / total if total else 0.0,
        "num_identities": int(subset["identity_id"].nunique()),
        "pairing_counts": {k: int(v) for k, v in pairing_counts.items()},
    }


def assert_both_classes_present(stats: Dict[str, object]) -> None:
    """Hard guard: a single-class subset is untrainable (BCE has nothing to
    separate) and its AUC is undefined, so fail immediately rather than
    burning hours of preprocessing and training to discover it."""
    if stats["num_real"] == 0 or stats["num_fake"] == 0:
        raise ValueError(
            f"Subset collapsed to a single class ({stats['num_real']} real, "
            f"{stats['num_fake']} fake). Check subset_size / subset_real_fraction, "
            "and that the indexed manifest actually contains both classes."
        )


def warn_if_too_few_real(
    stats: Dict[str, object], min_real: int = 200
) -> Optional[str]:
    """Returns a warning string if the subset has too few real clips for
    downstream metrics to be trustworthy, else None."""
    if stats["num_real"] < min_real:
        return (
            f"Only {stats['num_real']} real clips in the subset (<{min_real}). After "
            "identity-disjoint splitting this may leave too few real samples per split "
            "for a stable AUC or a meaningful false-suppression rate. Consider raising "
            "target_size or target_real_fraction."
        )
    return None
