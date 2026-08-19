import torch
from torch.utils.data import Dataset

from src.models.fusion.late_fusion import average_probabilities, predict_branch_probabilities


def test_average_probabilities_plain_average():
    visual = [0.9, 0.1, 0.8]
    acoustic = [0.7, 0.3, 0.2]
    fused = average_probabilities(visual, acoustic, visual_weight=0.5)
    assert fused == [0.8, 0.2, 0.5]


def test_average_probabilities_weighted():
    visual = [1.0, 0.0]
    acoustic = [0.0, 1.0]
    fused = average_probabilities(visual, acoustic, visual_weight=0.75)
    assert fused == [0.75, 0.25]


def test_average_probabilities_length_mismatch_raises():
    try:
        average_probabilities([0.1, 0.2], [0.5])
        assert False, "expected AssertionError"
    except AssertionError:
        pass


class _DummyDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        x = torch.full((3,), self.values[idx])
        mask = torch.zeros(3, dtype=torch.bool)
        label = torch.tensor(0.0)
        return x, mask, label


class _SumModel(torch.nn.Module):
    def forward(self, x, padding_mask=None):
        return x.sum(dim=-1)


def test_predict_branch_probabilities_matches_manual_sigmoid():
    dataset = _DummyDataset([0.0, 10.0, -10.0])
    model = _SumModel()

    probs = predict_branch_probabilities(model, dataset, torch.device("cpu"), batch_size=2)

    expected = [torch.sigmoid(torch.tensor(v * 3)).item() for v in [0.0, 10.0, -10.0]]
    assert len(probs) == 3
    for got, want in zip(probs, expected):
        assert abs(got - want) < 1e-5
