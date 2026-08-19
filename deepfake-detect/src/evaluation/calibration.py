"""Stage 6, part 1: post-hoc temperature scaling for the frozen Stage 5
fusion model. T is fit on the CALIBRATION split (never val/train) by
minimizing NLL; c = 1 - sigmoid(logit / T) is the calibrated authenticity
score used downstream by the decision policy (src/evaluation/policy.py).
"""

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # headless: this runs in training/eval scripts, not a GUI session
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from scipy.special import expit
from torch.utils.data import DataLoader


def get_fusion_logits(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs the frozen fusion model over `dataset` and returns (raw fused
    logits, integer labels) in split order -- pre-sigmoid, since
    temperature scaling operates on logits, not probabilities."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()

    logits, labels = [], []
    with torch.no_grad():
        for frames, v_mask, features, a_mask, batch_labels in loader:
            outputs = model(
                frames.to(device), v_mask.to(device), features.to(device), a_mask.to(device)
            )
            logits.extend(outputs["y_hat_logit"].cpu().tolist())
            labels.extend(batch_labels.tolist())

    return np.array(logits, dtype=np.float64), np.array(labels, dtype=np.int64)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> float:
    """Fits a single scalar T minimizing BCE-with-logits(logits/T, labels)
    via LBFGS, optimizing in log-space so T stays positive. Standard
    temperature-scaling recipe (Guo et al., 2017)."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)
    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).item())


def calibrated_fake_probability(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """sigmoid(logit / T) = P(fake). Uses scipy's expit rather than a
    hand-rolled 1/(1+exp(-x)), which overflows for large-magnitude logits
    (a confident model easily produces |logit| > 700)."""
    return expit(logits / temperature)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """c = 1 - sigmoid(logit / T): the calibrated authenticity score."""
    return 1.0 - calibrated_fake_probability(logits, temperature)


def compute_reliability_diagram(
    fake_probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict:
    """Standard reliability diagram: bins samples by predicted P(fake), and
    for each bin reports (mean predicted probability, empirical fraction
    that really is fake, sample count). Also returns Expected Calibration
    Error (ECE), the sample-weighted mean gap between the two."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(fake_probabilities, bin_edges[1:-1]), 0, n_bins - 1)

    bin_confidence, bin_accuracy, bin_count = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        bin_count.append(count)
        if count > 0:
            bin_confidence.append(float(fake_probabilities[mask].mean()))
            bin_accuracy.append(float(labels[mask].mean()))
        else:
            bin_confidence.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
            bin_accuracy.append(0.0)

    total = max(len(fake_probabilities), 1)
    ece = float(
        sum(
            (bin_count[b] / total) * abs(bin_confidence[b] - bin_accuracy[b])
            for b in range(n_bins)
        )
    )

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_count": bin_count,
        "ece": ece,
    }


def plot_reliability_diagram(before: dict, after: dict, output_path: str) -> None:
    """Side-by-side reliability diagrams (predicted P(fake) vs. empirical
    fraction fake) before and after temperature scaling, so miscalibration
    and its correction are visible at a glance."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, diagram, title in [(axes[0], before, "Before calibration"),
                                (axes[1], after, "After calibration")]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
        ax.plot(diagram["bin_confidence"], diagram["bin_accuracy"], marker="o", color="tab:blue")
        ax.set_title(f"{title} (ECE={diagram['ece']:.4f})")
        ax.set_xlabel("Predicted P(fake)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("Empirical fraction fake")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
