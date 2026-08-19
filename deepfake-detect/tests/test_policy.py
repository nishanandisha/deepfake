import numpy as np

from src.evaluation.policy import (
    decide,
    detection_recall,
    false_suppression_rate,
    resolve_thresholds,
    review_queue_rate,
    select_thresholds,
)


def test_decide_boundaries():
    assert decide(0.9, tau_lo=0.2, tau_hi=0.8) == "approve"
    assert decide(0.8, tau_lo=0.2, tau_hi=0.8) == "approve"  # >= tau_hi
    assert decide(0.5, tau_lo=0.2, tau_hi=0.8) == "flag"
    assert decide(0.2, tau_lo=0.2, tau_hi=0.8) == "flag"  # >= tau_lo -> not blocked
    assert decide(0.1, tau_lo=0.2, tau_hi=0.8) == "block"


def test_false_suppression_rate_counts_only_real_blocked():
    c_scores = np.array([0.9, 0.1, 0.05, 0.9])
    labels = np.array([0, 0, 1, 1])  # real, real, fake, fake
    # tau_lo=0.2 blocks the second real sample (0.1) -> 1 of 2 real blocked
    assert false_suppression_rate(c_scores, labels, tau_lo=0.2) == 0.5


def test_review_queue_rate_counts_flag_band():
    c_scores = np.array([0.9, 0.5, 0.3, 0.1])
    rate = review_queue_rate(c_scores, tau_lo=0.2, tau_hi=0.8)
    assert rate == 0.5  # 0.5 and 0.3 fall in [0.2, 0.8)


def test_detection_recall_counts_fakes_not_approved():
    c_scores = np.array([0.9, 0.5, 0.1])
    labels = np.array([0, 1, 1])
    recall = detection_recall(c_scores, labels, tau_hi=0.8)
    assert recall == 1.0  # both fakes (0.5, 0.1) are below tau_hi


def test_select_thresholds_respects_false_suppression_ceiling():
    rng = np.random.default_rng(0)
    real_scores = np.clip(rng.normal(0.9, 0.05, 200), 0, 1)
    fake_scores = np.clip(rng.normal(0.1, 0.05, 200), 0, 1)
    c_scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([np.zeros(200), np.ones(200)])

    result = select_thresholds(c_scores, labels, false_suppression_ceiling=0.02, grid_size=200)

    assert result["false_suppression_rate"] <= 0.02
    assert result["detection_recall"] > 0.8  # well-separated distributions -> easy to catch fakes


def test_resolve_thresholds_manual_override():
    c_scores = np.array([0.9, 0.5, 0.1, 0.4])
    labels = np.array([0, 1, 1, 0])

    result = resolve_thresholds(c_scores, labels, tau_hi=0.7, tau_lo=0.3)

    assert result["tau_hi"] == 0.7
    assert result["tau_lo"] == 0.3
    assert "false_suppression_rate" in result
    assert "detection_recall" in result
