"""The shipped artefact is the whole point of this package, so it gets its
own tests: it must load, rebuild, and score a real clip end to end.

These skip (rather than fail) when models/acoustic_model.joblib is absent,
so a fresh clone that hasn't fetched the weights still has a green suite.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference.loader import DEFAULT_MODEL_PATH, audio_settings_from, load_acoustic_model

REPO = Path(__file__).resolve().parents[1]
ARTEFACT = Path(DEFAULT_MODEL_PATH)
SAMPLES = sorted((REPO / "samples").rglob("*.mp4"))

requires_artefact = pytest.mark.skipif(
    not ARTEFACT.exists(), reason="models/acoustic_model.joblib not present"
)


@requires_artefact
def test_artefact_rebuilds_and_loads_strictly():
    loaded = load_acoustic_model(str(ARTEFACT))
    assert loaded.metadata["dataset"]
    assert loaded.input_dim == 3 * loaded.n_mfcc + 8  # mfcc+delta+delta2 + 8 scalars
    assert sum(p.numel() for p in loaded.model.parameters()) == \
        loaded.metadata["num_parameters"]


@requires_artefact
def test_artefact_is_the_acoustic_model_only():
    """Guard against someone dropping the visual or fusion artefact in here:
    this package ships the audio branch and nothing else."""
    import joblib

    assert joblib.load(str(ARTEFACT))["name"] == "acoustic"
    assert not list((REPO / "models").glob("visual_model.joblib"))
    assert not list((REPO / "models").glob("fusion_model.joblib"))


@requires_artefact
def test_manifest_digest_matches_the_file_on_disk():
    import hashlib

    manifest = json.loads((REPO / "models" / "manifest.json").read_text())
    assert [m["name"] for m in manifest["models"]] == ["acoustic"]

    h = hashlib.sha256()
    with open(ARTEFACT, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == manifest["models"][0]["sha256"]


@requires_artefact
def test_forward_pass_is_finite_and_mask_aware():
    loaded = load_acoustic_model(str(ARTEFACT))
    settings = audio_settings_from(loaded.data_config)
    frames = settings["num_audio_frames"]

    features = torch.randn(2, frames, loaded.input_dim)
    mask = torch.zeros(2, frames, dtype=torch.bool)
    mask[1, frames // 2 :] = True

    with torch.no_grad():
        logits = loaded.model(features, padding_mask=mask)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


@requires_artefact
def test_padding_does_not_change_the_score():
    """Padded frames are masked out, so extending a clip with padding must
    leave the logit alone. If this drifts, every score on a short clip is
    quietly wrong."""
    loaded = load_acoustic_model(str(ARTEFACT))
    torch.manual_seed(0)
    valid = 120
    features = torch.randn(1, valid, loaded.input_dim)

    padded = torch.cat([features, torch.zeros(1, 80, loaded.input_dim)], dim=1)
    mask = torch.zeros(1, valid + 80, dtype=torch.bool)
    mask[0, valid:] = True

    with torch.no_grad():
        short = loaded.model(features, padding_mask=torch.zeros(1, valid, dtype=torch.bool))
        long = loaded.model(padded, padding_mask=mask)

    assert torch.allclose(short, long, atol=1e-4)


@pytest.mark.slow
@requires_artefact
@pytest.mark.skipif(not SAMPLES, reason="no sample clips bundled")
def test_end_to_end_inference_on_a_sample_clip():
    from src.inference.pipeline import AudioDeepfakePipeline, InferenceConfig

    # SHAP off: this test is about the pipeline wiring, and KernelSHAP would
    # dominate the runtime.
    pipeline = AudioDeepfakePipeline(InferenceConfig(explain=False, device="cpu"))
    result = pipeline.infer(str(SAMPLES[0]))

    assert result["modality"] == "audio"
    assert 0.0 <= result["pFake"] <= 1.0
    assert np.isclose(result["cScore"], 1.0 - result["pFake"], atol=1e-3)
    assert result["decision"] in {"approve", "flag", "block"}
    assert result["calibrated"] is False  # no policy shipped -- see models/README.md
    assert result["framesAvailable"] > 0
