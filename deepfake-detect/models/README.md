# Trained models

Serialised with `joblib` so each file is self-contained: weights, the
architecture config needed to rebuild the graph, and the metrics of the run
that produced it. Regenerate with `python scripts/export_models.py`, and
check them with `python scripts/verify_models.py` (non-zero exit on failure).

| Artefact | Params | Size | Test AUC |
|---|---|---|---|
| `visual_model.joblib` | 4,448,958 | 15.99 MB | 0.976 |
| `acoustic_model.joblib` | 286,410 | 1.02 MB | 0.972 |
| `fusion_model.joblib` | 4,917,644 | 17.65 MB | 0.993 |

Trained on LAV-DF. Metrics are test-split, from `scripts/evaluate_test.py`.
`manifest.json` carries SHA-256 digests and export timestamps.

## Contents of an artefact

```python
{
  "name": "fusion",
  "framework": "pytorch",
  "torch_version": "...",
  "weights":     OrderedDict,  # state_dict, on CPU
  "config":      dict,         # rebuilds the graph
  "data_config": dict,         # frame rate/size, audio framing, n_mfcc
  "metadata":    dict,         # test metrics, calibration policy, sha256, ...
}
```

Weights are stored on CPU, so these load on a machine with no GPU.

## Loading

`config` is stored in the exact shape its builder expects, so no reshaping is
needed. Note the asymmetry: the two standalone builders take the model section
itself, while `build_fusion_model_for_inference` reads `cfg.model.*` and so
gets the enclosing wrapper.

```python
import joblib, torch
from omegaconf import OmegaConf
from src.training.train_fusion import build_fusion_model_for_inference

art   = joblib.load("models/fusion_model.joblib")
model = build_fusion_model_for_inference(OmegaConf.create(art["config"]))
model.load_state_dict(art["weights"])
model.eval()
```

For the single-modality branches, swap in `build_visual_model` or
`build_acoustic_model` from `src.training.train_visual` / `train_acoustic`.

## Fine-tuning

These are inference-ready weights, not resumable training states — the
optimizer state is deliberately not included (it roughly triples the file size
and is only needed to continue the *same* run, never to start a new one). So
build a fresh optimizer:

```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # lower than the 1e-4 used to train
```

To fine-tune only the fusion head and leave the pretrained branches intact:

```python
for p in model.visual_encoder.parameters():   p.requires_grad = False
for p in model.acoustic_encoder.parameters(): p.requires_grad = False
```

Two things worth carrying over from how these were trained:

- **Class balancing is applied once, not twice.** The training loop picks
  either a weighted sampler or a `pos_weight` on the loss — applying both
  double-corrects and collapses the model to a single class. See
  `make_pos_weight()` in `src/training/common.py`.
- **Keep the capacity where it is.** The encoders were deliberately sized down
  (`embed_dim=128`, depth 2) after a larger configuration collapsed to a
  constant function on this data. Scaling back up needs proportionally more
  data. `RESULTS.md` has the diagnosis.

## Calibration

`fusion_model.joblib` carries the fitted decision policy in
`metadata["calibration_policy"]`: the temperature plus the `tau_lo` / `tau_hi`
thresholds for the approve / flag / block bands. Raw sigmoid outputs are
uncalibrated — apply the temperature before thresholding.
