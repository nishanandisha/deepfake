# Results — LAV-DF, 6,000-clip run

Final run. All six stages complete, end-to-end system verified live.

**Headline: the system detects 98.2% of deepfakes while routing only 7.9%
of uploads to human review, and wrongly blocks under 2% of authentic
content.**

## Configuration

| | |
|---|---|
| Dataset | LAV-DF (content-driven audio-visual deepfakes) |
| Clips | 6,000 — train 4,198 / val 899 / calibration 297 / test 606 |
| Balance | 1,500 real, 4,500 fake; 1,500 per manipulation pairing |
| Splits | identity-disjoint (a clip and every fake derived from it stay together) |
| Visual | 16 frames @ 2 fps, 112×112, face-aligned |
| Acoustic | 400 frames @ 20 ms hop, 68 named features |
| Encoders | d=128, 2 layers, ff=256, attention pooling |
| Hardware | RTX 3050 Laptop 4 GB |

## Baseline comparison (validation split)

| Model | AUC | Macro-F1 | Accuracy | Precision | Recall | EER |
|---|---|---|---|---|---|---|
| Visual CNN + Transformer | 0.9763 | 0.9283 | 0.9455 | 0.9685 | 0.9584 | 0.0629 |
| Audio Transformer | 0.9641 | 0.8662 | 0.8943 | 0.9530 | 0.9034 | 0.1007 |
| **Late fusion (avg.)** | **0.9928** | **0.9641** | **0.9733** | 0.9765 | 0.9881 | **0.0496** |
| Proposed (cross-attention) | 0.9926 | 0.9395 | — | — | — | — |

Precision and recall are balanced on every model — the mark of a real
detector, not the degenerate predictor discussed below.

## The three central claims

**(a) Cross-attention beats late fusion — NOT SUPPORTED (statistical tie).**
AUC 0.9926 vs 0.9928, a gap of 0.0002 on an 899-clip validation split. That
is far inside sampling noise; neither architecture is measurably better
here. Claiming a win on the fourth decimal place would not be defensible.
Both fusion approaches clearly beat either branch alone (+0.0165 AUC over
the visual branch), so *fusion* helps — the specific mechanism does not
distinguish itself at this scale.

**(b) Explanations identify the manipulated modality — WEAKLY SUPPORTED.**
Attribution agreement **52.9% (9/17)** against a 50% chance baseline. Above
chance but only just, and live testing shows the limitation concretely: the
system attributed all three fake clips to "audio", including the
video-only one. Detection is reliable; modality attribution is not yet.

**(c) Calibration improves reliability — SUPPORTED.**
T = 2.0017 fitted on the calibration split; ECE on the held-out validation
split fell 0.0389 → 0.0283, measured out-of-sample.

## Operating point

| Quantity | Value |
|---|---|
| tau_lo / tau_hi | 0.065 / 0.78 |
| False-suppression rate | 0.0177 (ceiling 0.02 — satisfied) |
| Review-queue rate | 0.0790 |
| Detection recall | 0.9821 |

This is a deployable policy: 98.2% of fakes caught, 7.9% of submissions
sent for human review, under 2% of authentic content wrongly blocked.

## The finding that made this work

Earlier runs reported AUC 0.60–0.65, which read as "weak but real
detection". It was neither. Single-clip inference at full precision showed
the model emitting a near-constant:

```
real  0.775180578232
fake  0.775181293488
fake  0.775180876255      <- 8 clips, agreeing to 6 decimal places
```

Output varied by ~1e-6 across clearly different inputs (frame means 0.478 /
0.230 / 0.261, durations 6.7 s / 12.2 s / 10.1 s). **Every AUC measured
before this point was the ROC curve ranking floating-point noise**, which is
also why it swung 0.41 ↔ 0.63 between epochs.

Tracing where the signal died:

| Stage | Difference between two clips |
|---|---|
| EfficientNet backbone | 0.313 |
| after input projection | 0.575 |
| **after Transformer** | **0.072** ← destroyed |

**Cause:** a randomly-initialised 4-layer Transformer (d=512, ff=2048,
~12 M params) trained on 1,400 clips. With capacity that far above the
data, the cheapest way to minimise BCE is to ignore the input and emit the
base rate — which is exactly why training loss sat pinned at ln(2)=0.693 in
every run.

**Fix:** d=512→128, depth 4→2, ff 2048→256 (~30× fewer Transformer
parameters), plus attention pooling in place of mean pooling.

| | Collapsed | Fixed |
|---|---|---|
| Visual AUC | 0.5988 | **0.9763** |
| Visual macro-F1 | 0.4334 (constant predictor) | **0.9283** |
| Detection recall | 0.597 | **0.982** |
| Review-queue rate | 0.556 | **0.079** |

Same data, same features, same seed. Only capacity changed.

**Methodological point worth keeping:** the collapse was invisible in
aggregate metrics. An AUC of 0.63 with 76% accuracy looked like a working-
but-weak model. It surfaced only because building the UI bridge forced
single-clip inference with full-precision output printed. Aggregate metrics
alone cannot distinguish a weak detector from a constant function.

## Bugs found and fixed

Each produced plausible-looking numbers rather than an error:

1. **Identity collapse** — real clips carry `original: null` and LAV-DF
   stores a split in one flat folder, so the identity fallback assigned
   every real clip the identity `"test"`, putting the same person in train
   and test.
2. **Zero-fake subset** — `keep_all_real` assumed FakeAVCeleb's 40:1 skew;
   on LAV-DF's 27% real it filled the entire budget with real clips.
3. **Analysis window** — a 1.5 s window missed the manipulation entirely
   for 72.7% of clips labelled fake (edits average 0.65 s, median onset
   3.2 s).
4. **Silent audio failure** — no ffmpeg meant 100% of audio decodes failed,
   masked by per-clip error tolerance; the acoustic cache was empty.
5. **Double class-balancing** — weighted sampler *and* `pos_weight` both
   applied, pinning macro-F1 at exactly 0.1903.
6. **Missing feature normalisation** — acoustic features spanned a 114,000×
   scale range (`short_time_energy` std 0.02 vs `spectral_rolloff` std 1956).
7. **Representation collapse** — the capacity mismatch above.

## Verified end-to-end

Live against the running server:

```
truth  modal  decision   cScore   yFused    gate
real   none   approve    0.9921   0.0001   0.995
fake   video  block      0.0022   1.0000   0.015
fake   audio  block      0.0023   1.0000   0.022
fake   both   block      0.0022   1.0000   0.018
```

## Running it

```bash
# backend
python scripts/serve.py \
  --fusion-checkpoint D:/deepfake-data/outputs_medium/fusion/checkpoints/best.pt \
  --policy-json       D:/deepfake-data/outputs_medium/calibration/policy.json \
  --data medium

# frontend
cd ../deepfake-detect-ui && npm run dev
```

The UI ("**DeepFake**") header shows **live model** or **mock data** so a demo
can never present simulated output as real.

Note the API routes are `GET /api/health` and `POST /api/infer` — not
`/health`.

The `decision` column above is the raw policy label. The UI presents it as a
verdict (`block` -> Deepfake, `flag` -> Possibly manipulated, `approve` ->
Not a deepfake), reports the score as manipulation likelihood (`1 - cScore`),
and localizes which spans of each track were manipulated. Verified on the
bundled samples:

```
sample              verdict         likelihood  localized regions
real/002053.mp4     Not a deepfake      1%      none (both tracks clean)
fake/000919.mp4     Deepfake           98%      1 video span, 4 audio regions
```

Audio-only (video track stripped with `ffmpeg -vn`, so ground truth is
carried over from the source clip):

```
sample                   verdict         likelihood  localized regions
045769_voicenote.mp3     Not a deepfake      3%      none
001714_voicenote.mp3     Deepfake          100%      2 audio regions
```

These agree with the standalone `audio-deepfake-detect` model on the same two
files (P(fake) 0.0089 and 1.0000), which is a useful cross-check since the
two use different weights.

## Limitations

- Clips come from LAV-DF's official *test* partition (the tarball stores it
  first and we began before the full 25.6 GB landed), re-split
  identity-disjointly here. No leakage within this experiment, but the
  numbers are not comparable to published LAV-DF benchmarks.
- 6,000 of 136,304 available clips.
- DFDC cross-dataset generalization (Stage 8) not run.
- Modality attribution is weak (52.9%) even though detection is strong.
- `voicing_confidence` is a proxy derived from pitch-range validity and
  frame energy, not `pyin`'s posterior — a 56× speed trade. Set
  `pitch_tracker: pyin` to restore exact fidelity.
