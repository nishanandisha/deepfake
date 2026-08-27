# Trained model — audio branch only

One artefact: `acoustic_model.joblib`. The visual and fusion models of the
parent multimodal project are deliberately **not** here — this package
detects manipulated audio and nothing else.

Serialised with `joblib` so the file is self-contained: weights, the
architecture config needed to rebuild the graph, the data config it was
trained under, and the metrics of the run that produced it. Check it with
`python scripts/verify_model.py` (non-zero exit on failure).

| Artefact | Params | Size | Test AUC |
|---|---|---|---|
| `acoustic_model.joblib` | 286,410 | 1.02 MB | 0.972 |

Trained on LAV-DF. Metrics are test-split. `manifest.json` carries the
SHA-256 digest and the export timestamp; the digest is identical to the one
in the parent project's manifest, so this is byte-for-byte the same trained
model, not a re-export.

Full test metrics as embedded in the artefact:

| Metric | Value |
|---|---|
| accuracy | 0.9142 |
| precision | 0.9725 |
| recall | 0.9138 |
| macro F1 | 0.8878 |
| AUC | 0.9723 (95% CI 0.9611–0.9823) |
| EER | 0.0924 |

## Contents of the artefact

```python
{
  "name": "acoustic",
  "framework": "pytorch",
  "torch_version": "...",
  "weights":     OrderedDict,  # state_dict, on CPU
  "config":      dict,         # rebuilds the graph
  "data_config": dict,         # sample rate, framing, n_mfcc, pitch tracker
  "metadata":    dict,         # test metrics, sha256, provenance
}
```

Weights are stored on CPU, so this loads on a machine with no GPU.

## Loading

```python
from src.inference.loader import load_acoustic_model

loaded = load_acoustic_model("models/acoustic_model.joblib")
logits = loaded.model(features, padding_mask=mask)   # features [B, S, 68]
```

`load_acoustic_model` rebuilds the graph from the embedded `config` and
loads the weights with `strict=True`, so a config/weights mismatch fails at
load rather than silently producing wrong numbers. `loaded.data_config`
carries the framing the model expects — feed it differently-framed features
and you still get a number, just a number about the wrong input.

The raw equivalent, if you would rather not use the helper:

```python
import joblib
from omegaconf import OmegaConf
from src.training.train_acoustic import build_acoustic_model

art   = joblib.load("models/acoustic_model.joblib")
model = build_acoustic_model(OmegaConf.create(art["config"]))
model.load_state_dict(art["weights"])
model.eval()
```

## Fine-tuning

These are inference-ready weights, not a resumable training state — the
optimizer state is deliberately not included (it roughly triples the file
size and is only needed to continue the *same* run, never to start a new
one). Build a fresh optimizer:

```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # lower than the 3e-4 used to train
```

Two things worth carrying over from how this was trained:

- **Class balancing is applied once, not twice.** The training loop picks
  either a weighted sampler or a `pos_weight` on the loss — applying both
  double-corrects and collapses the model to a single class. See
  `make_pos_weight()` in `src/training/common.py`.
- **Keep the capacity where it is.** The encoder was deliberately sized down
  (`embed_dim=128`, depth 2) after a larger configuration collapsed to a
  constant function on this data. Scaling back up needs proportionally more
  data. The reasoning is in `configs/model/acoustic.yaml`.

## Calibration

This artefact carries **no** calibration policy — `metadata["calibration_policy"]`
is empty by design. The parent project fitted temperature and the
approve/flag/block thresholds on the *fused* audio-visual logit, and those
numbers do not transfer to this model's differently-scaled logit.

Raw sigmoid outputs are uncalibrated. To get an operating point with a
measured false-suppression rate, fit an audio-only policy against your own
calibration split:

```
python scripts/calibrate.py --config-name calibration
python scripts/predict.py clip.wav --policy outputs/calibration/policy.json
```
