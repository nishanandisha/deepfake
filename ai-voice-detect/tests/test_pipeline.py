"""Guards on the parts that would fail silently."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predict import voiced_seconds
from src.models.head import VoiceClassifierHead
from src.preprocessing.dataset import collate_padded
from src.preprocessing.manifest import _parse_stem
from src.preprocessing.splits import assert_no_group_leakage, make_splits
from src.training.metrics import compute_eer


def test_parse_stem_maps_prefix_to_generator():
    assert _parse_stem("el_0001_c_part_002") == ("elevenlabs", "el_0001")
    assert _parse_stem("yt_0000_p2_part_167") == ("youtube", "yt_0000")


def test_parse_stem_rejects_unknown_prefix():
    # An unrecognised generator must fail loudly rather than silently become
    # its own pseudo-source and quietly corrupt the holdout.
    with pytest.raises(ValueError):
        _parse_stem("zz_0001_part_001")


def _toy_manifest() -> pd.DataFrame:
    rows = []
    for source, prefix, label in [("polly", "po", 1), ("hume", "hu", 1), ("youtube", "yt", 0)]:
        for group_id in range(6):
            for part in range(4):
                rows.append({
                    "path": f"/tmp/{prefix}_{group_id:04d}_part_{part}.flac",
                    "label": label, "label_name": "fake" if label else "real",
                    "source": source, "group": f"{prefix}_{group_id:04d}",
                })
    return pd.DataFrame(rows)


def test_splits_never_share_a_group():
    splits = make_splits(_toy_manifest(), held_out_sources=["hume"], seed=0)
    assert_no_group_leakage(splits)


def test_held_out_generator_appears_only_in_test():
    splits = make_splits(_toy_manifest(), held_out_sources=["hume"], seed=0)
    assert set(splits[splits["source"] == "hume"]["split"]) == {"test"}


def test_eer_is_zero_for_perfect_separation():
    labels = [0, 0, 0, 1, 1, 1]
    assert compute_eer(labels, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) == pytest.approx(0.0)


def test_eer_is_half_when_scores_are_reversed():
    labels = [0, 0, 0, 1, 1, 1]
    assert compute_eer(labels, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]) == pytest.approx(1.0)


def test_collate_masks_padding_not_content():
    batch = [(torch.ones(10, 768), torch.tensor(1.0)),
             (torch.ones(4, 768), torch.tensor(0.0))]
    features, mask, labels = collate_padded(batch)
    assert features.shape == (2, 10, 768)
    assert not mask[0].any()          # longest clip: no padding
    assert mask[1][4:].all()          # short clip: tail is padding
    assert not mask[1][:4].any()      # ...but its real frames are not masked
    assert labels.tolist() == [1.0, 0.0]


def test_head_ignores_padded_frames():
    """Padding must not change the score, or short clips get a different
    verdict purely from what they were batched with."""
    torch.manual_seed(0)
    model = VoiceClassifierHead().eval()
    real = torch.randn(1, 12, 768)

    padded = torch.cat([real, torch.randn(1, 20, 768)], dim=1)
    mask = torch.zeros(1, 32, dtype=torch.bool)
    mask[:, 12:] = True

    with torch.no_grad():
        a = model(real, torch.zeros(1, 12, dtype=torch.bool))
        b = model(padded, mask)
    assert torch.allclose(a, b, atol=1e-4)


def test_silence_reports_no_voiced_audio():
    """The exact failure that made the predecessor score silence 0.98 fake."""
    assert voiced_seconds(np.zeros(16000 * 10, dtype=np.float32)) == 0.0


def test_speech_like_signal_reports_voiced_audio():
    rng = np.random.default_rng(0)
    tone = (np.sin(2 * np.pi * 200 * np.arange(16000 * 2) / 16000)
            * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * np.arange(16000 * 2) / 16000)))
    signal = (tone + 0.01 * rng.standard_normal(16000 * 2)).astype(np.float32)
    assert voiced_seconds(signal) > 0.5
