# Audio deepfake detection

> ## ⚠️ Superseded by [`../ai-voice-detect`](../ai-voice-detect)
>
> This package no longer serves any request. Its 68 hand-crafted MFCC-family
> descriptors proved to be the accuracy ceiling rather than the training
> budget: mel-binning followed by DCT truncation discards phase and fine
> spectral structure — exactly where TTS artifacts live — leaving mostly
> recording-condition information behind.
>
> Measured failures that motivated the replacement:
>
> | probe | result |
> |---|---|
> | 10s of digital silence | scored **0.98 "fake"** |
> | macOS `say`, 4 voices | 3/4 caught, only ever `flag`, one missed at 0.0034 |
> | cross-dataset generalisation | never run — see `../deepfake-detect/RESULTS.md:199` |
>
> The replacement pairs a frozen WavLM-Base+ frontend with a small trained
> head: **2.00% EER** on a held-out split including two TTS generators withheld
> from training, abstains on non-speech, windows long files, and fits its own
> audio-only calibration instead of borrowing the fused model's.
>
> Kept for reference, and for the SHAP-over-named-features story that a
> learned-embedding model cannot provide.

The audio half of the multimodal deepfake detector in `../deepfake-detect`,
extracted so it stands on its own: one trained model, one modality, no video
code path and no OpenCV dependency.

Give it an audio file — or a video container whose audio is muxed in, the
video track is never decoded — and it returns a manipulation probability plus
a SHAP attribution over named acoustic descriptors ("F0 and the MFCC deltas
drove this decision"), not an opaque score.

```
python scripts/predict.py samples/fake/000919.mp4
```

```
000919.mp4
  P(fake)        0.9999   -> FAKE
  authenticity c 0.0001   decision: block  (UNCALIBRATED -- plain 0.5 cut)
  audio analysed 308 frames (6.14s of signal)
  top acoustic features (SHAP, + pushes toward fake):
           mfcc_delta2_0  +6.61060  ##############################
                  mfcc_0  +5.83085  ##########################
                      f0  -5.12286  #######################
```

## The model

`models/acoustic_model.joblib` — 286,410 parameters, 1.02 MB, **test AUC
0.972** (95% CI 0.961–0.982) on LAV-DF. This is the only model in this
package; the visual and fusion artefacts are deliberately absent. Its SHA-256
is identical to the one in the parent project's manifest, so it is
byte-for-byte the same trained weights, not a re-export. See
[models/README.md](models/README.md) for the full metrics table, the artefact
schema, and fine-tuning notes.

Architecture: 68 hand-crafted per-frame descriptors (20 MFCCs + deltas +
delta-deltas, F0, voicing confidence, four spectral statistics, ZCR, energy)
→ per-feature BatchNorm → linear projection to d=128 → sinusoidal positional
encoding → 2-layer Transformer encoder (4 heads) → attention pooling → MLP
head → one logit.

Three design decisions worth knowing, each documented at its call site:

- **Hand-crafted features, not a learned embedding.** Every input dimension
  has a human-readable name, which is the only reason the SHAP output is
  interpretable at all (`src/models/acoustic/features.py`).
- **The encoder is small on purpose.** A 4-layer d=512 version collapsed to a
  constant function on this much data — logits agreeing to six decimal places
  across different clips. The reasoning is in `configs/model/acoustic.yaml`;
  scale it back up only alongside more data.
- **Attention pooling, not mean pooling.** LAV-DF manipulations average 0.65s
  inside ~8.5s clips, so averaging dilutes the evidence to ~8% of the pooled
  vector (`src/models/common.py`).

## Install and check

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/verify_model.py   # rebuilds the graph, loads weights strictly, forward pass
python -m pytest                 # 97 tests
```

`verify_model.py` reads *only* the joblib file — never a training checkpoint
or the Hydra configs — so a config/weights mismatch fails there rather than
silently producing wrong numbers later.

## Scoring clips

```bash
python scripts/predict.py clip.wav
python scripts/predict.py samples/real/*.mp4 --no-explain --json out.json
python scripts/predict.py clip.wav --policy outputs/calibration/policy.json
```

`--no-explain` skips KernelSHAP, which dominates the runtime. From Python:

```python
from src.inference.pipeline import AudioDeepfakePipeline, InferenceConfig

pipeline = AudioDeepfakePipeline(InferenceConfig())   # load once
result = pipeline.infer("clip.wav")                    # serve many
```

The pipeline takes its framing parameters (sample rate, window, hop, frame
count, pitch tracker) from the artefact's own `data_config` by default.
Overriding them is possible and almost always wrong: differently-framed
features still produce a number, just a number about the wrong input.

### About that "UNCALIBRATED" note

The shipped artefact carries **no** decision policy. The parent project
fitted temperature and the approve/flag/block thresholds on the *fused*
audio-visual logit, and those numbers do not transfer to this model's
differently-scaled logit — so rather than reuse them and quietly misreport
the operating point, this package ships without one.

Until you fit your own, `cScore` is the raw `1 - sigmoid(logit)` and the
decision is a plain 0.5 cut with no measured false-suppression rate. To get a
real operating point:

```bash
python scripts/calibrate.py --config-name calibration
python scripts/predict.py clip.wav --policy outputs/calibration/policy.json
```

That fits temperature on the calibration split, then searches thresholds on
validation subject to a hard ceiling on the false-suppression rate (authentic
audio wrongly blocked) — never on test.

## Training from scratch

Needs the LAV-DF dataset on disk; point `data.root_dir` at it.

```bash
python scripts/build_splits.py     # identity-disjoint train/val/calibration/test
python scripts/warm_cache.py       # extract features once, not once per epoch
python scripts/train.py            # or: data=lean training=lean
python scripts/evaluate.py         # test-split metrics + bootstrap AUC CI
python scripts/export_model.py --checkpoint outputs/acoustic_branch/checkpoints/best.pt \
                               --test-results outputs/acoustic_branch/test_results.json
```

Splits are identity-disjoint by construction and the splitter fails loudly on
leakage — a fake and the real clip it was derived from always land in the same
split, or the model gets scored on speakers it trained on.

`data=lean` is the laptop preset: 2,000 clips, a 20ms hop, 8s of coverage.
Every config file explains its own tradeoffs.

## Layout

```
models/            the trained artefact + manifest + loading docs
configs/           Hydra configs (data / model / training presets)
src/
  models/acoustic/ feature extraction and the Transformer classifier
  preprocessing/   audio decode, framing, dataset, cache, splits, subsetting
  training/        training loop, checkpointing, metrics
  evaluation/      calibration, decision policy, results writing
  explain/         KernelSHAP over the named features
  inference/       artefact loader + the single-clip pipeline
scripts/           the CLI entry points listed above
samples/           12 LAV-DF clips (6 real, 6 fake) for a smoke test
```

## Limits worth stating

- **Clip-level only.** The model says whether a clip contains manipulated
  audio, not *where*. The `loudRegions` in the pipeline output are an energy
  heuristic for display, explicitly not a localization — don't present them
  as one.
- **One dataset.** All metrics are LAV-DF. Cross-dataset generalisation is
  unmeasured here.
- **`voicing_confidence` is a proxy.** The default `yin` pitch tracker is
  ~177x faster than `pyin` but has no voicing posterior, so that feature is
  derived from pitch-range plausibility and frame energy. Switch to
  `pitch_tracker=pyin` if it ranks high in your SHAP output and you need the
  true posterior.
