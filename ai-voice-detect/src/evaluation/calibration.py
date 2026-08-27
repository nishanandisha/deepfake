"""Temperature scaling and threshold selection.

A trained head emits logits whose *ranking* is good but whose *magnitude*
means nothing -- a 0.97 does not indicate 97% confidence. Temperature scaling
(Guo et al., 2017) fits one scalar on held-out data so the numbers the user
sees can be read as probabilities.

One scalar is the entire point: it cannot reorder any pair of clips, so EER
and AUC are unchanged by construction. It only fixes the scale, and being
one parameter it fits reliably on the small calibration split this corpus
can afford.

Fitted on the dedicated `calibration` split, never on train (already fit,
so temperature would collapse toward 1) and never on test (that would make
the reported numbers self-graded).
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from src.training.metrics import compute_eer


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Scalar T minimising NLL of sigmoid(logit / T)."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)

    def nll(log_t: float) -> float:
        temperature = np.exp(log_t)  # keeps T > 0 without a constrained solver
        probabilities = np.clip(expit(logits / temperature), 1e-7, 1 - 1e-7)
        return float(
            -np.mean(labels * np.log(probabilities)
                     + (1 - labels) * np.log(1 - probabilities))
        )

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated P(AI-generated)."""
    return expit(np.asarray(logits, dtype=float) / temperature)


def select_threshold(
    probabilities: np.ndarray, labels: np.ndarray, target: str = "eer"
) -> float:
    """Operating threshold on calibrated probabilities.

    `eer` balances the two error types, which is the right default when
    neither mistake is obviously worse. `f1` is offered for the case where
    missing AI audio matters more than annoying a human speaker.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    candidates = np.unique(np.round(probabilities, 4))

    best_threshold, best_cost = 0.5, float("inf")
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        n_neg = max(int((labels == 0).sum()), 1)
        n_pos = max(int((labels == 1).sum()), 1)

        if target == "eer":
            cost = abs(fp / n_neg - fn / n_pos)
        else:
            tp = int(((predictions == 1) & (labels == 1)).sum())
            cost = -(2 * tp / max(2 * tp + fp + fn, 1))

        if cost < best_cost:
            best_threshold, best_cost = float(threshold), cost

    return best_threshold


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """Mean gap between confidence and accuracy, weighted by bin population."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    edges = np.linspace(0.0, 1.0, bins + 1)

    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities > low) & (probabilities <= high)
        if not mask.any():
            continue
        # Confidence must be in the *predicted* class, not in class 1:
        # a clip called human at p=0.02 is 98% confident, not 2%. Using p
        # directly scores every correct negative as maximally miscalibrated.
        predictions = (probabilities[mask] >= 0.5).astype(int)
        confidence = np.where(predictions == 1,
                              probabilities[mask], 1 - probabilities[mask]).mean()
        accuracy = (predictions == labels[mask]).mean()
        error += mask.mean() * abs(confidence - accuracy)
    return float(error)


def build_policy(
    logits: np.ndarray, labels: np.ndarray, target: str = "eer"
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit temperature and threshold, reporting ECE before and after."""
    raw = expit(np.asarray(logits, dtype=float))
    temperature = fit_temperature(logits, labels)
    calibrated = apply_temperature(logits, temperature)
    threshold = select_threshold(calibrated, labels, target=target)

    policy = {
        "temperature": temperature,
        "threshold": threshold,
        "calibration_eer": compute_eer(labels, calibrated),
        "ece_before": expected_calibration_error(raw, labels),
        "ece_after": expected_calibration_error(calibrated, labels),
        "n_calibration": int(len(labels)),
    }
    return policy, calibrated


def save_policy(policy: Dict[str, float], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2))


def load_policy(path: str) -> Dict[str, float]:
    return json.loads(Path(path).read_text())
