"""Binary metrics, with EER as the headline.

EER is the convention in speech anti-spoofing and is reported instead of
accuracy because it is threshold-free: accuracy on a corpus that happens to
be 50/50 flatters a model that would fall apart at any other operating point,
and the deployed threshold is chosen later by `calibrate.py` anyway.
"""

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Equal error rate: the point where FPR and FNR coincide.

    Interpolated rather than taken at the nearest ROC vertex, which matters
    on small test sets where the curve is coarse.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[index] + fnr[index]) / 2)


def compute_metrics(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    both_classes = len(np.unique(y_true)) > 1
    return {
        "eer": compute_eer(y_true, y_score),
        "auc": float(roc_auc_score(y_true, y_score)) if both_classes else float("nan"),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "n": int(len(y_true)),
        "n_ai": int((y_true == 1).sum()),
        "n_human": int((y_true == 0).sum()),
    }


def per_source_metrics(
    y_true: Sequence[int], y_score: Sequence[float], sources: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Per-generator breakdown.

    A single pooled EER hides the case that matters: strong average numbers
    carried by the easy generators while one unseen system is missed
    outright. Human rows are included in every generator's slice so each
    figure is a real detection rate, not a one-class score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    sources = np.asarray(sources)

    human = y_true == 0
    out: Dict[str, Dict[str, float]] = {}
    for source in sorted(set(sources[y_true == 1])):
        mask = human | (sources == source)
        out[source] = compute_metrics(y_true[mask], y_score[mask])
    return out
