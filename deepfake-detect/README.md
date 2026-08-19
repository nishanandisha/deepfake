# deepfake-detect

Explainable multimodal (audio-visual) deepfake detection: identity-disjoint
FakeAVCeleb training with cross-dataset (DFDC) generalization testing, a
calibrated three-way decision policy, and SHAP/Grad-CAM explanations for
moderator review.

See [`deepfake-detection-build-plan.md`](../deepfake-detection-build-plan.md)
(one level up) for the full 9-stage build plan and exit criteria for each
stage. This repo is built stage by stage; **do not skip ahead** — each branch
must be validated standalone before fusion, per the plan's golden rule.

## Setup

**Requires Python 3.11 or 3.12** (see `.python-version`). Python 3.13/3.14
are too new for this stack as of writing -- Hydra 1.3.4 crashes on 3.14's
stricter `argparse`, and `opencv-python` 5.0 dropped the `CascadeClassifier`
face-detector binding this repo relies on. If you only have a newer Python
installed, get 3.12 via `uv python install 3.12` (or pyenv/python.org) and
point the venv at it explicitly, as below.

```bash
# Windows, using a uv-installed 3.12:
"$(uv python find 3.12)" -m venv .venv
.venv\Scripts\activate
# source .venv/bin/activate   # macOS/Linux

python scripts/dev.py setup   # or: make setup (if you have make)
python scripts/dev.py test    # or: make test
```

Windows has no `make` by default, so every Makefile target has an equivalent
`python scripts/dev.py <target>` command (see the Makefile for the list).

GPU: `requirements.txt` pins `torch`/`torchvision`/`torchaudio` to the
`+cu128` builds via `--extra-index-url`. If your machine doesn't have an
NVIDIA GPU, drop that index line and the `+cu128` suffixes to get CPU-only
wheels instead.

## Configuration system

**Hydra** (+ OmegaConf), not plain YAML + argparse. Reasoning: Stage 8's
ablation study needs to run the same fusion model with one component
swapped at a time (minus temporal transformer, minus cross-attention, minus
gating, minus aux losses), and Stage 5 needs `model=visual|acoustic|fusion`
style composition. Hydra gives config composition and CLI overrides
(`python scripts/train.py model=visual training.lr=1e-4`) for free instead of
hand-rolling override-merging logic. Structure:

```
configs/
  config.yaml           # top-level: seed, output dir, defaults list
  data/default.yaml      # dataset paths, preprocessing, augmentation params
  training/default.yaml  # optimizer, schedule, early stopping
  model/visual.yaml       # visual branch hyperparams
  model/acoustic.yaml     # acoustic branch hyperparams
  model/fusion.yaml       # composes visual + acoustic + cross-attn/gate/aux-loss params
```

## Pipeline stages

The system is built and validated in six stages, each independently testable
before the next depends on it:

1. **Preprocessing** — face detection/alignment + frame sampling for video;
   resampling + framing for audio. Produces identity-disjoint train / val /
   calibration / test splits (no identity ever appears in more than one
   split) so evaluation numbers reflect generalization, not memorized faces.
2. **Visual branch** — CNN backbone (per-frame spatial features) + Transformer
   encoder (temporal relationships across frames), trained and validated
   standalone with its own auxiliary classification head.
3. **Acoustic branch** — named, human-readable features (MFCC + deltas, F0,
   spectral stats, ZCR/energy) + Transformer encoder, trained and validated
   standalone. Kept feature-based (not a learned black-box embedding)
   specifically so SHAP can attribute the decision to individual descriptors
   later.
4. **Cross-modal fusion** — bidirectional cross-attention between the visual
   and acoustic streams, a learned gate mixing the two into one fused vector,
   and auxiliary unimodal losses so each branch keeps its own signal instead
   of collapsing onto the other. Compared directly against a late-fusion
   (probability-averaging) baseline — that comparison is the project's
   central empirical claim.
5. **Decision policy** — post-hoc temperature calibration (fit on a
   calibration split distinct from val/test) followed by a three-way
   approve/flag/block threshold policy, tuned so the false-suppression rate
   (real content wrongly blocked) stays under a configurable ceiling.
6. **Explanation** — coarse SHAP over the visual/acoustic modality split,
   fine-grained SHAP over the named acoustic features, and Grad-CAM saliency
   over implicated video frames, assembled into a single moderator-facing
   report per sample.

## Repo layout

```
configs/          YAML configs per experiment (Hydra)
data/             dataset download + caching scripts (not raw data)
src/
  preprocessing/  face/audio processing, split generation
  models/
    visual/       CNN + Transformer encoder
    acoustic/     named-feature extractor + Transformer encoder
    fusion/       cross-attention + gating
  training/       training loops per branch + joint fusion training
  evaluation/     metrics, calibration, decision policy
  explain/        SHAP + Grad-CAM + report generation
scripts/          CLI entry points (train.py, eval.py, infer.py, dev.py)
notebooks/        exploratory analysis
tests/
```

## Data

`scripts/build_splits.py` indexes FakeAVCeleb into `data/splits/{train,val,calibration,test}.csv`
(identity-disjoint, see `src/preprocessing/splits.py`) and DFDC into
`data/splits/dfdc_holdout.csv` (indexed only, never split or trained on).
Both datasets are gated and must be obtained manually:

- **FakeAVCeleb**: request access from the dataset authors, then set
  `data.root_dir` (in `configs/data/default.yaml` or via CLI override) to
  the extracted root containing `RealVideo-RealAudio/`, `FakeVideo-FakeAudio/`,
  etc.
- **DFDC**: download via Kaggle (full set or the smaller sample set), then
  set `data.dfdc_root_dir` to the folder containing `dfdc_train_part_*/`.

```bash
python scripts/build_splits.py data.root_dir=/path/to/FakeAVCeleb data.dfdc_root_dir=/path/to/dfdc
```

The FakeAVCeleb directory-layout assumptions in `src/preprocessing/manifest.py`
(identity folder depth, category folder names) are documented in that file's
docstring but have not yet been verified against a real downloaded copy of
the dataset -- check `identity_dir_depth` in the config if indexing finds 0
samples or misgroups identities.

## Status

**Complete and working.** All six pipeline stages run end to end on real
LAV-DF data, and the Next.js UI is connected to the trained model.

| Model | AUC | Macro-F1 | Accuracy | EER |
|---|---|---|---|---|
| Visual CNN + Transformer | 0.9763 | 0.9283 | 0.9455 | 0.0629 |
| Audio Transformer | 0.9641 | 0.8662 | 0.8943 | 0.1007 |
| **Late fusion (avg.)** | **0.9928** | **0.9641** | **0.9733** | **0.0496** |
| Cross-attention fusion | 0.9926 | 0.9395 | — | — |

Operating point: **98.2% detection recall, 7.9% review-queue rate, 1.77%
false suppression** (under the 2% ceiling).

See **[RESULTS.md](RESULTS.md)** for the full analysis — including the
representation-collapse finding that took the visual branch from AUC 0.5988
to 0.9763, and the seven bugs that each produced plausible-looking numbers
rather than errors.

Not done: Stage 8 ablations and the DFDC cross-dataset test.

Note the dataset changed from the original plan: FakeAVCeleb requires
manual approval from its authors, whereas LAV-DF is reachable through a
HuggingFace click-through licence and labels `modify_video`/`modify_audio`
separately — which is what the attribution-agreement metric scores against.

Stage-by-stage implementation notes follow.

- **Stage 0** — scaffolding.
- **Stage 1** — indexing, identity-disjoint splitting, preprocessing,
  augmentation, class-imbalance handling.
- **Stage 2** — visual branch: EfficientNet-B0 backbone + sinusoidal
  positional encoding + Transformer encoder (`src/models/visual/encoder.py`),
  standalone training loop with weighted sampling, AdamW + cosine schedule,
  early stopping on val AUC, and the 5 standard metrics
  (`src/training/train_visual.py`). Run via
  `python scripts/train.py model=visual`.
- **Stage 3** — acoustic branch: named MFCC/delta/delta-delta + F0 +
  voicing-confidence + spectral-stats + ZCR/energy feature extraction
  (`src/models/acoustic/features.py`, `feature_names` ordering verified
  against tensor columns by test) + linear projection + Transformer encoder
  (`src/models/acoustic/encoder.py`), trained via the same standalone loop
  (`src/training/train_acoustic.py`, `python scripts/train.py model=acoustic`).
  The shared training engine (sampler/loader/optimizer/schedule/early-stop/
  checkpoint/results-md) was factored out into `src/training/common.py` once
  Stage 3 needed the identical logic Stage 2 already had.

Both branches' training-loop sanity tests (`tests/test_train_{visual,acoustic}_overfit.py`)
confirm the mechanics are correct by overfitting a tiny synthetic dataset to
AUC > 0.9 — this is **not** a substitute for the real Stage 2/3 exit criteria
(AUC clearly above chance on real FakeAVCeleb val data), which still needs
the real dataset. One thing to watch once real data lands: in both sanity
runs, AUC reached 1.0 while default-threshold accuracy lagged behind —
the raw 0.5 decision threshold is uncalibrated until Stage 6.

- **Stage 4** — late-fusion baseline: loads the frozen Stage 2/3 checkpoints
  (never modifies them) and averages each branch's standalone probability
  (`src/models/fusion/late_fusion.py`, evaluated via
  `src/evaluation/late_fusion_eval.py` /
  `python scripts/evaluate_late_fusion.py`). This surfaced a real config
  bug worth knowing about: `configs/model/fusion.yaml` originally merged
  `model/visual.yaml` and `model/acoustic.yaml` as flat Hydra defaults,
  which silently collided on their shared `transformer`/`embed_dim` keys.
  Fixed by namespacing them (`visual@visual`, `acoustic@acoustic`), so
  `cfg.model.visual.*` and `cfg.model.acoustic.*` stay independent — keep
  using that nested form for Stage 5's fusion config, not a flat merge.
- **Stage 5** — cross-modal attention fusion, the core contribution:
  bidirectional cross-attention (`Hv` attends to `Ha` and vice versa) +
  residual/LayerNorm + a learned gate mixing the two attended, pooled
  streams into one fused vector, plus auxiliary unimodal losses
  (`src/models/fusion/cross_attention.py`). Trained jointly via
  `src/training/train_fusion.py` / `python scripts/train.py model=fusion`
  (also the default model if `model=` is omitted). Decision per the build
  plan: both branches warm-start from their frozen Stage 2/3 checkpoints,
  then fine-tune jointly rather than staying frozen — the auxiliary losses
  exist specifically to stop either branch collapsing onto the other during
  that fine-tuning (`cfg.init_from_standalone_checkpoints`, default `true`).
  The gate's per-sample visual-vs-acoustic weighting is logged every epoch
  (`avg_gate_visual_weight`) so modality collapse would be visible
  immediately. After training, the results markdown automatically loads
  Stage 4's `late_fusion_results` (a JSON sidecar every stage's
  `write_results_markdown` call now writes) and reports whether
  cross-attention actually beat late fusion — the build plan's central
  empirical claim — with real numbers, or "not available" if Stage 4 hasn't
  been run yet in that `outputs/` tree.
- A `MultimodalDataset` (`src/preprocessing/dataset.py`) was added to pair
  visual + acoustic samples in one batch, needed once Stage 5 required both
  modalities simultaneously for the joint forward pass (Stage 4 didn't need
  this since it evaluates each frozen branch independently and then
  averages).

- **Stage 6** — calibration & decision policy: temperature scaling fit by
  LBFGS on the **calibration** split (never val/test), then approve/flag/block
  threshold selection on val subject to a hard false-suppression ceiling
  (default 2%), with a before/after reliability diagram and ECE
  (`src/evaluation/calibration.py`, `src/evaluation/policy.py`, run via
  `python scripts/calibrate.py`). Writes `policy.json` (temperature +
  thresholds) for Stage 7/8 to consume.

Three bugs worth remembering, all found by tests rather than review:

1. **Checkpointing froze on plateaus.** All three training loops used a
   strict `auc > best_auc` comparison, so once AUC plateaued (trivially, at
   1.0 on synthetic data) the checkpoint stuck at the *first, least-trained*
   epoch that reached it — discarding later, better-calibrated weights.
   Ties now break on validation loss.
2. **Sigmoid overflow.** `1/(1+np.exp(-logit/T))` overflows once
   `|logit| > ~700`, which a confident model reaches easily. Replaced with
   `scipy.special.expit`.
3. **Order-dependent test flakiness.** The long-running overfit/e2e tests
   never seeded RNG, so weight init depended on which tests ran before them
   — the Stage 6 test passed alone but failed in the full suite. All five
   now call `set_seed` explicitly. **Keep doing this for any new test that
   asserts on learned values.**

No model logic beyond the two standalone branches + late fusion + joint
cross-attention fusion + calibration/policy (Stage 7 onward: explanation
layer) implemented yet.
