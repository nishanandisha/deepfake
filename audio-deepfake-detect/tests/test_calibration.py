from pathlib import Path

import numpy as np
import torch

from src.evaluation.calibration import (
    apply_temperature,
    compute_reliability_diagram,
    fit_temperature,
    get_acoustic_logits,
    plot_reliability_diagram,
)


def test_apply_temperature_matches_manual_sigmoid():
    logits = np.array([-2.0, 0.0, 2.0])
    temperature = 2.0
    expected = 1.0 - 1.0 / (1.0 + np.exp(-logits / temperature))
    assert np.allclose(apply_temperature(logits, temperature), expected)


def test_fit_temperature_corrects_overconfident_logits():
    # Genuine overconfidence (not just noise): the model claims near-100%
    # confidence via large-magnitude logits, but is only actually correct
    # 75% of the time -- the classic case temperature scaling fixes.
    # (Perfectly-separable logits, by contrast, are already NLL-optimal at
    # low T; softening them with T>1 would only increase loss.)
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=1000)
    is_correct = rng.random(1000) < 0.75
    predicted_label = np.where(is_correct, labels, 1 - labels)
    overconfident_logits = np.where(predicted_label == 1, 8.0, -8.0) + rng.normal(0, 0.1, size=1000)

    temperature = fit_temperature(overconfident_logits, labels)

    assert temperature > 3.0  # should scale confidence down substantially

    criterion = torch.nn.BCEWithLogitsLoss()
    logits_t = torch.tensor(overconfident_logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    nll_before = criterion(logits_t, labels_t).item()
    nll_after = criterion(logits_t / temperature, labels_t).item()
    assert nll_after < nll_before


def test_reliability_diagram_near_zero_ece_when_well_calibrated():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, size=2000)
    labels = (rng.uniform(0, 1, size=2000) < probs).astype(int)

    diagram = compute_reliability_diagram(probs, labels, n_bins=10)
    assert diagram["ece"] < 0.1


def test_reliability_diagram_high_ece_when_miscalibrated():
    rng = np.random.default_rng(2)
    probs = np.full(1000, 0.9)  # always confident...
    labels = rng.integers(0, 2, size=1000)  # ...but actually 50/50

    diagram = compute_reliability_diagram(probs, labels, n_bins=10)
    assert diagram["ece"] > 0.3


def test_plot_reliability_diagram_writes_file(tmp_path: Path):
    labels = np.array([1] * 50 + [0] * 50)
    before = compute_reliability_diagram(np.array([0.9] * 50 + [0.1] * 50), labels)
    after = compute_reliability_diagram(np.array([0.6] * 50 + [0.4] * 50), labels)

    output_path = tmp_path / "reliability.png"
    plot_reliability_diagram(before, after, str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


class _DummyAcousticModel(torch.nn.Module):
    def forward(self, features, padding_mask=None):
        # Just needs to be a deterministic function of the input.
        return features.mean(dim=(1, 2)) * 10


class _DummyAcousticDataset(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        value = 1.0 if self.labels[idx] == 1 else 0.0
        features = torch.full((3, 5), value)
        mask = torch.zeros(3, dtype=torch.bool)
        return features, mask, torch.tensor(float(self.labels[idx]))


def test_get_acoustic_logits_preserves_order_and_labels():
    labels = [0, 1, 0, 1, 1]
    dataset = _DummyAcousticDataset(labels)
    model = _DummyAcousticModel()

    logits, returned_labels = get_acoustic_logits(
        model, dataset, torch.device("cpu"), batch_size=2
    )

    assert len(logits) == 5
    assert list(returned_labels) == labels
    # Order preserved: label 1 -> all-ones features -> logit 10, label 0 -> 0.
    assert list(logits) == [0.0, 10.0, 0.0, 10.0, 10.0]
