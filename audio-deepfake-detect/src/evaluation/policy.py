"""Three-way approve/flag/block decision policy over the calibrated
authenticity score c = 1 - sigmoid(logit / T).

  approve: c >= tau_hi   (confidently authentic)
  flag:    tau_lo <= c < tau_hi   (uncertain -> moderator review queue)
  block:   c < tau_lo    (confidently manipulated)

Thresholds are searched on the VALIDATION split (never calibration/test)
subject to a hard constraint: false-suppression rate (authentic content
wrongly blocked) must stay at or under `false_suppression_ceiling`, since
wrongly blocking real citizen journalism is the costlier error for this
platform. Labels follow the project convention: 0 = real, 1 = fake.
"""

from typing import Dict, Optional

import numpy as np

DEFAULT_FALSE_SUPPRESSION_CEILING = 0.02


def decide(c_score: float, tau_lo: float, tau_hi: float) -> str:
    if c_score >= tau_hi:
        return "approve"
    if c_score < tau_lo:
        return "block"
    return "flag"


def false_suppression_rate(c_scores: np.ndarray, labels: np.ndarray, tau_lo: float) -> float:
    """Fraction of REAL samples that get blocked (the hard-constrained
    operational cost metric)."""
    real_mask = labels == 0
    if real_mask.sum() == 0:
        return 0.0
    blocked = c_scores < tau_lo
    return float((blocked & real_mask).sum() / real_mask.sum())


def review_queue_rate(c_scores: np.ndarray, tau_lo: float, tau_hi: float) -> float:
    """Fraction of ALL samples landing in the flag band (moderator workload
    proxy)."""
    flagged = (c_scores >= tau_lo) & (c_scores < tau_hi)
    return float(flagged.sum() / max(len(c_scores), 1))


def detection_recall(c_scores: np.ndarray, labels: np.ndarray, tau_hi: float) -> float:
    """Fraction of FAKE samples the policy does not wave through as
    "approve" (i.e. it gets flagged or blocked) -- the system's effective
    catch rate at this operating point."""
    fake_mask = labels == 1
    if fake_mask.sum() == 0:
        return 0.0
    not_approved = c_scores < tau_hi
    return float((not_approved & fake_mask).sum() / fake_mask.sum())


def operating_point_metrics(
    c_scores: np.ndarray, labels: np.ndarray, tau_lo: float, tau_hi: float
) -> Dict[str, float]:
    return {
        "false_suppression_rate": false_suppression_rate(c_scores, labels, tau_lo),
        "review_queue_rate": review_queue_rate(c_scores, tau_lo, tau_hi),
        "detection_recall": detection_recall(c_scores, labels, tau_hi),
    }


def select_thresholds(
    c_scores: np.ndarray,
    labels: np.ndarray,
    false_suppression_ceiling: float = DEFAULT_FALSE_SUPPRESSION_CEILING,
    grid_size: int = 200,
) -> Dict[str, float]:
    """Two-step greedy search over a [0,1] grid:
    1. tau_lo := the largest grid value with false_suppression_rate <=
       ceiling (falls back to 0.0 -- block nothing -- if even the smallest
       tau_lo violates the ceiling, which would only happen if real and
       fake score distributions are barely separated at all).
    2. tau_hi := the value in [tau_lo, 1] maximizing
       (detection_recall - review_queue_rate), a simple combined objective
       that rewards catching fakes and penalizes an oversized review queue.
       This tradeoff is a judgment call -- documented here so it can be
       revisited.
    """
    grid = np.linspace(0.0, 1.0, grid_size + 1)

    valid_tau_lo = [
        t for t in grid if false_suppression_rate(c_scores, labels, t) <= false_suppression_ceiling
    ]
    tau_lo = max(valid_tau_lo) if valid_tau_lo else 0.0

    best_tau_hi, best_score = tau_lo, -np.inf
    for t in grid[grid >= tau_lo]:
        score = detection_recall(c_scores, labels, t) - review_queue_rate(c_scores, tau_lo, t)
        if score > best_score:
            best_score, best_tau_hi = score, float(t)

    result = {"tau_lo": float(tau_lo), "tau_hi": best_tau_hi}
    result.update(operating_point_metrics(c_scores, labels, tau_lo, best_tau_hi))
    return result


def resolve_thresholds(
    c_scores: np.ndarray,
    labels: np.ndarray,
    tau_hi: Optional[float] = None,
    tau_lo: Optional[float] = None,
    false_suppression_ceiling: float = DEFAULT_FALSE_SUPPRESSION_CEILING,
    grid_size: int = 200,
) -> Dict[str, float]:
    """Uses manual tau_hi/tau_lo overrides if both are given, else runs
    select_thresholds. Either way, returns the operating-point metrics at
    the resulting thresholds."""
    if tau_hi is not None and tau_lo is not None:
        result = {"tau_lo": float(tau_lo), "tau_hi": float(tau_hi)}
        result.update(operating_point_metrics(c_scores, labels, tau_lo, tau_hi))
        return result
    return select_thresholds(c_scores, labels, false_suppression_ceiling, grid_size)
