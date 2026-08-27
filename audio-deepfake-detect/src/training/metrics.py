"""Shared classification metrics used by every branch's training/eval loop:
accuracy, precision, recall, macro-F1, AUC, EER.

AUC is the headline number (threshold-free ranking quality); EER gives the
single operating point where the two error types balance; macro-F1 is the
one that actually falls over when the model collapses onto a single class,
which accuracy alone hides on a skewed split.
"""

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Equal Error Rate: the point on the ROC curve where the false-positive
    rate equals the false-negative rate (1 - true-positive rate)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_binary_classification_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
        metrics["eer"] = compute_eer(y_true, y_prob)
    else:
        # AUC/EER are undefined with only one class present (e.g. a tiny
        # smoke-test batch) -- surface as NaN rather than raising.
        metrics["auc"] = float("nan")
        metrics["eer"] = float("nan")

    return metrics
