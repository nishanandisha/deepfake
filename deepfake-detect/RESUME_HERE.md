# Resume here

Paused 2026-08-06 ~00:55. Nothing is lost — both trained branches are on
disk and the pipeline picks up from where it stopped.

## One command to continue

```bash
cd "c:/Users/Admin/OneDrive/Desktop/major project/deepfake-detect"
./.venv/Scripts/python.exe -W ignore scripts/run_pipeline.py \
  --data medium --training lean \
  --output-root "D:/deepfake-data/outputs_medium" \
  --splits-dir  "D:/deepfake-data/splits_medium" \
  --cache-dir   "D:/deepfake-data/cache_medium" \
  --hydra-dir   "D:/deepfake-data/hydra_medium"
```

Visual and acoustic skip themselves (already at max epochs). It resumes at
late-fusion and runs the remaining four stages: **~50-70 minutes**.

## Done ✅

| Stage | AUC | Macro-F1 | Accuracy | EER |
|---|---|---|---|---|
| Visual CNN + Transformer | **0.9763** | 0.9283 | 0.9455 | 0.0629 |
| Audio Transformer | **0.9641** | 0.8662 | 0.8943 | 0.1007 |

Balanced precision/recall on both (0.968/0.958 and 0.953/0.903) — these are
real detectors, not the constant predictors the earlier runs produced.

## Remaining

- [ ] late-fusion baseline (~5 min)
- [ ] cross-attention fusion (~40-60 min) — the central claim
- [ ] calibration + policy (~4 min)
- [ ] explanations + attribution agreement (~5 min)
- [ ] rewrite RESULTS.md with the new numbers
- [ ] point the UI at the new checkpoint and demo it

## The finding that made this work

Every earlier run reported AUC 0.60-0.65 and I read it as "weak but real
detection". It was not. Single-clip full-precision inference showed the
model emitting a near-constant:

```
real  0.775180578232
fake  0.775181293488
fake  0.775180876255      <- 8 clips, agreeing to 6 decimal places
```

Output varied by ~1e-6 across clearly different inputs. Every AUC we had
measured was the ROC curve ranking floating-point noise, which is also why
it swung 0.41 <-> 0.63 between epochs.

**Cause:** a randomly-initialised 4-layer Transformer (d=512, ff=2048,
~12M params) on 1,400 clips. With capacity that far above the data, the
cheapest way to minimise BCE is to ignore the input and emit the base
rate — hence training loss pinned at exactly ln(2)=0.693 in every run.

**Fix:** d=512 -> 128, depth 4 -> 2, ff 2048 -> 256 (~30x fewer transformer
params). Visual AUC went 0.5988 -> 0.9763 with no other change: same data,
same features, same seed.

**Worth keeping in the report:** the collapse was invisible in aggregate
metrics. It only surfaced because building the UI bridge forced single-clip
inference with full precision printed. Constant-output models can produce
plausible-looking AUCs.

## Where things are

| What | Path |
|---|---|
| Code | `c:/Users/Admin/OneDrive/Desktop/major project/deepfake-detect` |
| UI | `c:/Users/Admin/OneDrive/Desktop/major project/deepfake-detect-ui` |
| Data + outputs | `D:/deepfake-data/` |
| Splits (6,000 clips) | `D:/deepfake-data/splits_medium/` |
| Cache (warm, 6000/6000) | `D:/deepfake-data/cache_medium/` |
| Sample clips to inspect | `D:/deepfake-data/INSPECT/` |

## To demo the UI once fusion finishes

```bash
# terminal 1 - backend
./.venv/Scripts/python.exe scripts/serve.py \
  --fusion-checkpoint D:/deepfake-data/outputs_medium/fusion/checkpoints/best.pt \
  --policy-json       D:/deepfake-data/outputs_medium/calibration/policy.json \
  --data medium

# terminal 2 - frontend
cd ../deepfake-detect-ui && npm run dev
```

The UI auto-detects the backend and falls back to its mock engine if the
server isn't up, so it always demos.

## Config notes (don't lose these)

- `batch_size: 4`, `num_workers: 2` — a 4GB card with Epic Games Launcher
  and Windows apps resident leaves ~3.2GB; 8 workers caused OOM.
- `expandable_segments` is a **no-op on Windows** (PyTorch warns and
  ignores). Headroom comes from batch size and worker count instead.
- `pitch_tracker: yin` — 56x faster than pyin; `voicing_confidence` is a
  proxy. Switch to `pyin` for a final high-fidelity run.
- Resume loads checkpoints to CPU first; `map_location=device` doubled VRAM
  and caused OOM on resume.
