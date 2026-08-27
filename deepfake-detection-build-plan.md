# Explainable Multimodal Deepfake Detection — Build Plan & Master Prompts

This document breaks the project into 9 stages, each with a self-contained "master prompt"
you can paste directly into Claude Code (or any coding agent) to build that stage. Each stage
also lists what "done" looks like, so you know when to move to the next one.

**Golden rule while building:** build and validate each branch in isolation before fusing.
If something breaks after fusion, you want to already know the visual branch and audio
branch work on their own.

---

## Stage 0 — Project scaffolding & environment

**Goal:** repo structure, dependency management, config system, reproducibility (seeds, logging).

**Master prompt:**
```
Set up a PyTorch project for a multimodal deepfake detection system. Create this structure:

deepfake-detect/
  configs/          # YAML configs per experiment
  data/             # dataset download + caching scripts (not raw data)
  src/
    preprocessing/
    models/
      visual/
      acoustic/
      fusion/
    training/
    evaluation/
    explain/
  scripts/          # CLI entry points (train.py, eval.py, infer.py)
  notebooks/
  tests/
  requirements.txt
  README.md

Requirements:
- Use Hydra or plain YAML + argparse for config management (pick one and justify briefly).
- Add a global seed-setting utility (torch, numpy, random, cuda) used at the start of every
  script for reproducibility.
- Set up logging (Python logging + optionally Weights & Biases or TensorBoard) with a single
  helper used across training scripts.
- Add a Makefile or scripts/ shortcuts for: setup, lint, test, train-visual, train-acoustic,
  train-fusion, evaluate.
- Write a README section describing the six pipeline stages (preprocessing, visual branch,
  acoustic branch, cross-modal fusion, decision policy, explanation) so future contributors
  understand the architecture before touching code.
- Add pytest with at least one placeholder test per module so CI has something to run.

Do not implement any model logic yet — this stage is scaffolding only.
```

**Exit criteria:** `make setup` installs cleanly, `pytest` runs (even if trivial), repo structure matches the above.

---

## Stage 1 — Data pipeline & identity-disjoint splits

**Goal:** download/organize FakeAVCeleb (primary) and DFDC (held-out generalization test), build preprocessing, and — critically — build identity-disjoint splits before any model code exists.

**Master prompt:**
```
Build the data pipeline for FakeAVCeleb and DFDC.

1. Write a loader that indexes FakeAVCeleb into a dataframe/manifest with columns:
   sample_id, video_path, audio_path, identity_id, label (real/fake),
   manipulated_modality (video/audio/both/none), source_generator (if available).

2. Implement identity-disjoint train/val/calibration/test splitting:
   - No identity_id may appear in more than one split.
   - Stratify splits so the real:fake ratio and the four
     (real-video/real-audio, fake-video/real-audio, real-video/fake-audio, fake-video/fake-audio)
     pairings are represented in train and val.
   - Write an assertion/test that fails loudly if any identity leaks across splits.
   - Output split manifests as CSVs under data/splits/.

3. Implement preprocessing:
   - Visual: sample frames at a fixed rate, run face detection + alignment, warp to 224x224,
     save as a frame tensor sequence per clip (or cache to disk/webdataset if the full dataset
     doesn't fit in memory).
   - Audio: resample to 16kHz mono, frame at 25ms window / 10ms hop.
   - Add augmentation ONLY on the train split: JPEG compression, gaussian noise, horizontal
     flip, random resized crop for video; additive noise + simple codec simulation for audio.

4. Implement a weighted sampler (or class-balanced loss weighting utility) to address the
   ~40:1 fake:real imbalance in FakeAVCeleb. Make it configurable so I can turn it on/off
   for ablation.

5. Write a small script that prints split statistics: sample counts, identity counts,
   real/fake ratio, and the four-way manipulation-pairing breakdown, per split.

Do not touch DFDC beyond indexing it — it must stay completely untouched by any training
or hyperparameter selection, reserved only for the final cross-dataset evaluation.
```

**Exit criteria:** split manifests exist, identity-leakage test passes, stats script confirms class balance strategy is working, DFDC is indexed but never referenced by training code.

---

## Stage 2 — Visual branch (standalone)

**Goal:** CNN spatial encoder + Transformer temporal encoder, trained and evaluated **on its own** against FakeAVCeleb val split, before fusion exists.

**Master prompt:**
```
Implement and train the visual-only branch of the deepfake detector.

Architecture:
- CNN backbone (start with a lightweight pretrained backbone, e.g. EfficientNet-B0 or a
  ResNet variant) mapping each aligned 224x224 frame to a spatial descriptor vector.
- Sinusoidal positional encoding over the frame sequence.
- Transformer encoder (configurable depth/heads, start with 4 layers, 8 heads) over the
  frame sequence, producing Hv of shape [T, d].
- Temporal mean pooling + small MLP classification head producing a manipulation
  probability for THIS BRANCH ALONE (used only for standalone validation and later as the
  auxiliary loss target during fusion training).

Training:
- Binary cross-entropy loss.
- Use the weighted sampler / loss weighting from Stage 1.
- AdamW, cosine LR schedule, early stopping on validation AUC.
- Log accuracy, precision, recall, macro-F1, AUC, EER on the validation split every epoch.

Deliverables:
- src/models/visual/encoder.py, src/training/train_visual.py
- A checkpoint saved to outputs/visual_branch/best.pt
- A short results markdown summarizing final visual-only metrics on val, to be used later
  as the "Visual CNN + Transformer" row in the baseline comparison table.

This branch must reach clearly-above-chance performance before we proceed — if AUC is near
0.5, stop and debug data/labels before moving to the acoustic branch.
```

**Exit criteria:** visual-only model trains without error, AUC noticeably above 0.5 on val, metrics logged and saved for later comparison.

---

## Stage 3 — Acoustic branch (standalone)

**Goal:** cepstral/prosodic/spectral feature extraction + Transformer, trained and evaluated on its own.

**Master prompt:**
```
Implement and train the acoustic-only branch of the deepfake detector.

Feature extraction per 25ms/10ms-hop frame:
- MFCCs (e.g. 13-20 coefficients) + delta + delta-delta.
- F0 (pitch) via a standard pitch tracker + voicing confidence.
- Spectral centroid, bandwidth, roll-off, flatness.
- Zero-crossing rate, short-time energy.
Concatenate into a named feature vector per frame — keep a feature_names list mapping each
dimension to a human-readable name; this will be required later for SHAP explanations, so
do not use a black-box learned embedding here.

Architecture:
- Linear projection of the feature vector to d dimensions.
- Transformer encoder (start 4 layers, 8 heads) over the frame sequence, producing
  Ha of shape [S, d].
- Temporal mean pooling + MLP classification head for standalone training/validation.

Training: same protocol as the visual branch (BCE loss, weighted sampling, AdamW + cosine
schedule, early stopping on val AUC, log accuracy/precision/recall/macro-F1/AUC/EER).

Deliverables:
- src/models/acoustic/features.py (with the named feature_names list),
  src/models/acoustic/encoder.py, src/training/train_acoustic.py
- Checkpoint at outputs/acoustic_branch/best.pt
- Results markdown for the "Audio Transformer" baseline row.

Stop and debug if AUC is near chance before moving on — this branch's features are exactly
what SHAP will explain later, so verify feature_names order matches the actual tensor
column order now, not after fusion is built.
```

**Exit criteria:** acoustic-only model trains, AUC above chance, feature_names verified against tensor columns, metrics saved.

---

## Stage 4 — Late-fusion baseline (quick, before cross-attention)

**Goal:** a trivial score-averaging baseline. This is cheap to build and gives you the exact comparison point the paper's central claim depends on.

**Master prompt:**
```
Using the trained visual and acoustic branch checkpoints from Stage 2 and 3, implement a
late-fusion baseline that averages (or learns a simple weighted average of) the two
branches' standalone probabilities. Evaluate it on the same validation split with the same
metrics (accuracy, precision, recall, macro-F1, AUC, EER). Save results as the
"Late fusion (avg.)" row in the baseline comparison table. Do not modify the visual or
acoustic branches for this stage — load their frozen checkpoints only.
```

**Exit criteria:** late-fusion metrics logged, ready to compare against cross-modal fusion later.

---

## Stage 5 — Cross-modal attention fusion (the core contribution)

**Goal:** bidirectional cross-attention + gated fusion + auxiliary unimodal losses, trained jointly (branches can be initialized from Stage 2/3 checkpoints or trained from scratch — decide and note which).

**Master prompt:**
```
Implement the cross-modal attention fusion module and joint training loop.

Architecture:
- Load Hv [T,d] from the visual branch and Ha [S,d] from the acoustic branch
  (reuse the encoder classes from Stage 2/3, not new copies).
- Bidirectional cross-attention:
    H_v_to_a = Attention(query=Hv, key=Ha, value=Ha)
    H_a_to_v = Attention(query=Ha, key=Hv, value=Hv)
- Residual connection + LayerNorm for each attended stream against its source.
- Temporal pooling of each attended stream into two vectors.
- A learned gate (small MLP producing a scalar or per-dim weight) that mixes the two
  pooled vectors into a single fused vector z. Log the gate's audio/visual weighting per
  sample — this becomes the "modality split" SHAP explains later, so expose it as a named,
  inspectable value now.
- MLP head mapping z to manipulation probability y_hat. Authenticity score c = 1 - y_hat.

Training objective:
  L = BCE(y_hat, label) + lambda_v * BCE(y_hat_visual, label) + lambda_a * BCE(y_hat_acoustic, label)
  where y_hat_visual / y_hat_acoustic come from each branch's own auxiliary head (reuse the
  standalone heads from Stage 2/3). Make lambda_v and lambda_a configurable; default both
  to a small positive value (e.g. 0.3) and note in the config that lambda=0 reproduces
  "no auxiliary loss" for the ablation study in Stage 8.

Train with the same optimizer/schedule/imbalance-handling as prior stages. Log the same
five metrics on validation every epoch, plus the average gate weighting to sanity-check
the model isn't collapsing onto one modality.

Deliverables: src/models/fusion/cross_attention.py, src/training/train_fusion.py,
checkpoint at outputs/fusion/best.pt, results markdown for the "Proposed" row.

After training, directly compare this run's F1/AUC against the Stage 4 late-fusion numbers
on the SAME validation split. This comparison is the paper's central empirical claim —
flag clearly in the results markdown whether cross-attention actually beats late fusion,
and by how much.
```

**Exit criteria:** fusion model trains without collapsing to one modality, metrics beat or are directly compared against late fusion, gate weights logged per sample.

---

## Stage 6 — Calibration & three-way decision policy

**Goal:** temperature scaling on a held-out calibration split (NOT val), then threshold selection for approve/flag/block.

**Master prompt:**
```
Implement post-hoc temperature calibration and the three-way decision policy.

1. Using the frozen fusion model from Stage 5, fit a single temperature parameter T by
   minimizing NLL on the CALIBRATION split (the split reserved in Stage 1, distinct from
   both train and val — do not reuse val for this).
2. Apply c = 1 - sigmoid(logit / T) as the calibrated authenticity score.
3. Implement threshold selection for approve (c >= tau_hi), flag (tau_lo <= c < tau_hi),
   block (c < tau_lo), searching thresholds on the validation split subject to a constraint:
   the false-suppression rate (authentic content wrongly blocked) must stay under a
   configurable ceiling (default 2%), since for this platform wrongly blocking real
   citizen journalism is treated as the more costly error.
4. Report, for the chosen thresholds: false suppression rate, review queue rate (% flagged),
   and detection recall at those operating points.
5. Add a reliability diagram (predicted probability vs empirical accuracy, before and after
   calibration) to the results markdown to visually confirm calibration improved.

Deliverables: src/evaluation/calibration.py, src/evaluation/policy.py, a config exposing
tau_hi/tau_lo/false-suppression-ceiling, and a results markdown with the reliability
diagram and operating-point table.
```

**Exit criteria:** reliability diagram shows improved calibration, thresholds chosen with false-suppression rate under ceiling, operating-point table produced.

---

## Stage 7 — Explanation layer (SHAP + saliency)

**Goal:** SHAP over acoustic features and the modality gate; CAM-based saliency over frames.

**Master prompt:**
```
Implement the explanation layer for the calibrated fusion model.

1. Coarse SHAP: attribute the fused decision between the pooled visual vector's
   contribution and the pooled acoustic vector's contribution (use the gate weighting from
   Stage 5 as the basis, and use KernelSHAP or a comparable method over these two
   "features" to get formal attribution values, not just the raw gate weight).
2. Fine SHAP: run SHAP (KernelSHAP is fine given ~20-40 named acoustic features) over the
   acoustic feature vector from Stage 3, using the feature_names list, to rank which
   descriptors (e.g. F0 variance, spectral flatness) drove the acoustic branch's
   contribution for a given sample.
3. Visual saliency: implement Grad-CAM (or a comparable CAM variant) over the visual
   branch's CNN backbone to highlight which spatial regions and frames most influenced the
   visual prediction. Do not attempt SHAP over raw pixels — use CAM here, per architecture
   design.
4. Build a simple report generator (script or notebook) that, given a sample, outputs:
   final score + decision, modality split (from step 1), top-k acoustic descriptors
   (from step 2) with their SHAP values, and saliency-overlaid frames (from step 3) for the
   most-implicated frames. This is the "moderator view" — even a static HTML/notebook
   output is enough for now, a full UI is optional.
5. Write a small evaluation: for a sample of manipulated_modality-labeled test clips
   (video-only fake, audio-only fake, both), check whether the coarse SHAP modality split
   agrees with the ground-truth manipulated modality. Report this as "attribution agreement
   rate" — this is your evidence that explanations are actually correct, not just present.

Deliverables: src/explain/shap_acoustic.py, src/explain/shap_modality.py,
src/explain/cam_visual.py, src/explain/report.py, and a results markdown with the
attribution agreement rate.
```

**Exit criteria:** report generator produces a readable explanation for a sample clip, attribution agreement rate computed and above a sane baseline (>50% at minimum — ideally much higher).

---

## Stage 8 — Full evaluation, ablation study, and cross-dataset test

**Goal:** assemble Table II (baseline comparison) and Table III (ablation) from the paper, plus the DFDC generalization check.

**Master prompt:**
```
Run the final evaluation suite and assemble results.

1. Baseline comparison table: re-run/evaluate Visual CNN only, Visual CNN + Transformer,
   Audio Transformer, Late fusion (avg.), and Proposed (cross-attention fusion) on the SAME
   held-out test split (not val, not calibration). Report accuracy, precision, recall,
   macro-F1, AUC for each.

2. Ablation table: using the Stage 5 fusion model, retrain/evaluate these configurations,
   changing exactly one thing each time:
   - Full model
   - minus temporal Transformer on the visual branch (replace with frame-level pooling)
   - minus cross-modal attention (replace with simple concatenation before the MLP head)
   - minus modality gating (replace gate with fixed 0.5/0.5 average)
   - minus auxiliary unimodal losses (set lambda_v = lambda_a = 0)
   Report F1 and AUC for each row.

3. Cross-dataset generalization: evaluate the frozen, calibrated Stage 6 model on DFDC
   (untouched until now) using the same metrics, plus report the accuracy/AUC drop
   relative to the FakeAVCeleb test split.

4. Operational metrics: report false-suppression rate and review-queue rate on the test
   split at the Stage 6 thresholds.

5. Attribution agreement rate from Stage 7, repeated on the test split.

Assemble everything into a single results.md replacing every "XX.XX" placeholder in the
paper draft with real numbers, and flag in writing whether each of the three central claims
holds: (a) cross-attention beats late fusion, (b) each ablated component contributes
positively, (c) cross-dataset degradation exists and is quantified.
```

**Exit criteria:** all three tables populated with real numbers, generalization gap reported, claims explicitly confirmed or flagged as unsupported.

---

## Stage 9 (optional, if time allows) — Moderator review interface

**Master prompt:**
```
Build a minimal moderator review interface (Streamlit or a small Flask/React app) that:
- Accepts a video+audio upload.
- Runs it through the full pipeline (preprocess -> branches -> fusion -> calibration ->
  policy -> explanation).
- Displays: decision (approve/flag/block), calibrated score, modality split, top acoustic
  descriptors with SHAP values, and saliency-overlaid frames.
Keep this stage isolated from the model/training code — it should only call a single
inference function (src/inference/pipeline.py) that wraps Stages 1-7 end to end.
```

**Built as:** `deepfake-detect-ui/` — a Next.js app called **DeepFake**,
talking to `deepfake-detect/scripts/serve.py` over `POST /api/infer`.

It diverges from the prompt above in two deliberate ways:

1. **It reports a verdict, not a moderation action.** `approve`/`flag`/`block`
   is a queue-routing label; the question users actually bring to the tool is
   "is this a deepfake?". The UI renders **Deepfake / Possibly manipulated /
   Not a deepfake**, driven by `cScore` against the calibrated thresholds, and
   states the score as manipulation likelihood so it reads in the same
   direction as the verdict. The backend contract is untouched.

2. **It localizes the manipulation.** A single clip-level score is not
   actionable when only a few seconds were altered. The UI derives timestamped
   spans per modality from the Grad-CAM salience and acoustic suspicious
   regions the pipeline already returns, draws them on a per-track timeline,
   and lets the reviewer confirm or reject each one individually before
   recording a final call.

   A branch must implicate its own modality (fake probability >= 0.5) before
   any of its regions are surfaced — salience is relative, and every clip has
   a most-salient frame, so without that gate the UI would invent findings on
   authentic clips.

---

## Evaluation metrics — decided

| Metric | Where used | Why |
|---|---|---|
| Accuracy | Baseline table (secondary) | Reported for comparability with prior work, but not trusted alone due to class imbalance |
| Precision / Recall | Baseline table | Standard detection quality |
| **Macro-F1** | Primary detection metric | Robust to the ~40:1 imbalance in FakeAVCeleb; use this over raw accuracy for all headline claims |
| **AUC** | Primary detection metric | Threshold-independent; used for early stopping and model selection |
| EER | Baseline table | Standard in the deepfake/anti-spoofing literature, makes results comparable to cited baselines |
| **False-suppression rate** | Policy evaluation | Operational cost metric — rate of authentic content wrongly blocked; has an explicit ceiling constraint (default 2%) |
| **Review-queue rate** | Policy evaluation | % of submissions landing in the flag band — proxy for moderator workload |
| **Attribution agreement rate** | Explanation evaluation | % of samples where the coarse SHAP modality split matches the ground-truth manipulated modality — this is what makes "explainable" a testable claim, not just a feature |
| Cross-dataset AUC/F1 drop (FakeAVCeleb → DFDC) | Generalization | Quantifies the known generalization gap in this literature; report explicitly rather than omitting it |

**Rules for reporting, decided now to avoid disputes later:**
- Every headline number is reported on the **test split only**, never on val (val is for model/threshold selection) and never on the calibration split.
- Macro-F1 and AUC are the two metrics used to decide "did cross-attention help" — not accuracy.
- False-suppression rate is treated as a hard constraint (ceiling), not just another metric to report — thresholds that violate it are rejected regardless of how good F1 looks.
- DFDC numbers are reported alongside FakeAVCeleb numbers in every final table, not as a footnote — the generalization gap is a stated limitation, not something to bury.
