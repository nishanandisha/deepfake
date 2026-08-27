# AI-generated voice detection

Binary classification of **human speech vs AI-generated speech**. One modality,
one question — no video, no deepfake-manipulation framing.

```bash
python scripts/predict.py clip.wav --layer 12
```

```
clip.wav
  P(AI)   0.9841  [#################################.]
  verdict AI-GENERATED
  audio   11.1s (9.4s voiced), 1 window(s), max 0.9841
```

## Results

Trained head: **0.40M parameters** over a frozen 94.4M WavLM-Base+ frontend.
Training took **288s** on an 8 GB M1.

| split | EER | AUC | accuracy | n |
|---|---|---|---|---|
| overall | **2.00%** | 0.9979 | 0.9747 | 356 |
| seen generators | 1.70% | 0.9990 | 0.9651 | 172 |
| **unseen generators** | **2.12%** | 0.9974 | 0.9680 | 281 |

Per generator — human rows included in each, so every figure is a real
detection rate rather than a one-class score:

| generator | EER | AUC | |
|---|---|---|---|
| ElevenLabs | 0.00% | 1.0000 | |
| Speechify | 0.00% | 1.0000 | |
| Kokoro | 1.25% | 0.9995 | **held out of training** |
| Polly | 1.81% | 0.9989 | |
| Hume | 3.79% | 0.9962 | **held out of training** |

Calibration: temperature 1.19, threshold 0.106, fit on 78 held-out clips.
ECE 0.062 → 0.073 (at that sample size the move is noise).

### External validation

Test-split numbers can flatter a model whose corpus has a shortcut, so the
model was also run on audio from outside the corpus entirely:

| input | result |
|---|---|
| macOS `say`, 6 voices — an engine **not in training** | **6/6 correct** |
| Real human screen recordings | **4/4 correct** |
| 10s digital silence | **abstains** |

### Reading these numbers honestly

`best val EER 0.45% at epoch 1` — the model solved the training distribution
in a single pass. Combined with a human class drawn from only 14 YouTube
videos, that is consistent with a shortcut: *"clean studio audio = AI,
compressed YouTube audio = human"*, which would transfer to any new TTS system
and make the small seen/unseen gap look like generalisation.

`scripts/shortcut_probe.py` tests this by pushing AI clips through the channel
degradations that characterise the human class:

```
AI clips   clean     mean P(AI) 0.8971   100.0% correct
           degraded  mean P(AI) 0.7338    95.0% correct
```

Synthesis evidence survives the codec, so it is not purely a channel detector.
That probe is weakened by the model having trained on that same augmentation —
the external validation above is the stronger evidence.

## Approach

**Frozen WavLM-Base+ → cached embeddings → a small trainable head.**

`torchaudio.pipelines.WAVLM_BASE_PLUS` (94.4M params, 768-dim, 12 layers,
16 kHz) runs **once** per clip and its output is written to disk. Training then
touches only the head, reading precomputed arrays.

| approach | per-epoch cost |
|---|---|
| **frozen + cached (this project)** | 0.4M params over `.npy` — seconds |
| LoRA / QLoRA | full forward+backward through 94M params, every epoch |
| full fine-tune | the above, plus 1.5 GB optimiser state |
| from scratch | hopeless on 2.17 hours of audio |

QLoRA is deliberately **not** used. It exists to fit billion-parameter LLMs
into limited VRAM; at 94M params 4-bit quantisation reclaims ~0.33 GB that was
never the bottleneck, `bitsandbytes` is CUDA-first with no usable MPS support,
and adapters cut optimiser state rather than the forward/backward pass that
dominates. Caching beats all of them on iteration speed because the expensive
pass happens exactly once.

**Why not MFCCs.** The predecessor (`../audio-deepfake-detect`) used 68
hand-crafted MFCC-family descriptors. Mel-binning plus DCT truncation discards
phase and fine spectral structure — exactly where TTS artifacts live. What
survives is largely recording-condition information, and it showed: that model
scored ten seconds of digital silence at **0.98 "fake"**.

## Data

[`garystafford/deepfake-audio-detection`](https://huggingface.co/datasets/garystafford/deepfake-audio-detection)
— CC-BY-4.0, ungated. 1,866 clips / 2.17 hours, 16 kHz mono FLAC (already
WavLM's native rate, so nothing is resampled on the way in).

| | clips | groups |
|---|---|---|
| human — YouTube | 933 | **14** |
| AI — Speechify | 211 | 10 |
| AI — Amazon Polly | 209 | 30 |
| AI — ElevenLabs | 173 | 28 |
| AI — Luvvoice | 156 | 6 |
| AI — Hume | 116 | 9 |
| AI — Hexgrad Kokoro | 68 | 13 |

**Use the parquet path.** Pulling 1,866 files individually gets rate-limited to
~1 file/sec, and a throttled `snapshot_download` exits **successfully** having
silently skipped an entire directory. The same data ships as two parquet shards
— two requests instead of 1,867:

```bash
python scripts/extract_parquet.py     # after fetching data/parquet/*.parquet
```

`scripts/download_data.py` is the file-by-file fallback; it retries and
verifies counts so a partial corpus cannot pass unnoticed.

> **Known limitation.** The 933 human clips are chunks of only **14 YouTube
> videos**. That is thin diversity for a whole class. Splitting constrains what
> the model can exploit; it does not eliminate it. Adding human speech from a
> non-YouTube source is the single highest-value improvement available.

## Splits

Two independent constraints, both violated by a naive random split:

- **Group disjointness** — every clip is a chunk of a longer recording, so all
  chunks of `yt_0000` live in exactly one split.
- **Generator holdout** — **Hume and Kokoro never appear in training.** A
  detector that has heard ElevenLabs is not thereby shown to detect synthetic
  speech; it may only detect ElevenLabs.

```
      split  clips  groups  human   ai  sources
      train   1178      39    654  524  elevenlabs,luvvoice,polly,speechify,youtube
        val    254      19    142  112  elevenlabs,polly,speechify,youtube
calibration     78      13     40   38  elevenlabs,polly,youtube
       test    356      39     97  259  + hume, kokoro  (held out)
```

`scripts/build_splits.py` asserts both properties on every run.

## Pipeline

```bash
python scripts/extract_parquet.py            # materialise real/ and fake/
python scripts/build_splits.py               # manifest + leakage assertions
python scripts/sweep_layers.py               # which WavLM layer to use
python scripts/cache_embeddings.py --layer 12   # the one expensive step (~20 min)
python scripts/train.py     --layer 12       # ~5 min
python scripts/calibrate.py --layer 12       # temperature + threshold
python scripts/evaluate.py  --layer 12       # seen vs UNSEEN generators
```

Only `cache_embeddings.py` touches WavLM. Retraining after that is a
minutes-long job, so head hyperparameters are cheap to explore.

**Layer choice.** The sweep found every layer scoring under 2% EER — on 254 val
clips, the 0.35% winner is about *one* error, so the layers are statistically
tied and the sweep cannot really discriminate. Layer 12 was taken as measured.
That near-tie is itself a signal that separating these classes on *seen*
generators is close to trivial.

## Serving

The model plugs into the existing UI at [`../deepfake-detect-ui`](../deepfake-detect-ui).
One command starts everything:

```bash
bash scripts/start_all.sh
```

```
UI :3000 -> router :8000 -+-> :8001  deepfake-detect  (visual branch, advisory)
                          +-> :8002  ai-voice-detect  (WavLM audio, verdict)
```

The UI only ever talks to port 8000. Three processes are an internal detail:
both projects ship a top-level package called `src`, so importing one shadows
the other, and they need different dependency sets (OpenCV/Hydra versus
torchaudio). Separate servers sidestep that cleanly.

Routing is **by probe, not extension** — an `.mp4` can carry no video track,
and an `.mp3` with cover art exposes the artwork as a single-frame video
stream. Two decodable frames are required before a file is treated as video.

### How the two branches combine

They answer genuinely different questions, and neither subsumes the other:

- **audio** — is this voice AI-generated?
- **visual** — was this face manipulated?

A face-swap over a real person's real voice is invisible to the audio model,
correctly so. Conversely, LAV-DF's "fakes" are *spliced real human speech*: the
audio model scores them 0.0001 while the visual branch scores 0.9929. So the
fused score is `max(visual, audio)` — a clip is suspect if either branch says
so — and `drivenBy` records which one set it.

> **The visual branch is advisory.** It is the old model, and it was measured
> scoring blank frames at 0.0039 against a real clip's 0.0038, while white,
> grey and noise frames all read 0.84–0.98 "fake". Its Haar face detector
> misses on 0–69% of frames even within its own training distribution, and
> fails silently by centre-cropping background. A visual-driven verdict is a
> prompt to review, not a conclusion. Only the audio branch has a validated
> number.

Individual servers, if needed: `scripts/serve.py` (audio only) and
`scripts/router.py`. Logs land in `outputs/logs/`.

## Design decisions

**Attentive statistics pooling.** A learned per-frame weighting concentrates on
frames carrying artifacts; the weighted *standard deviation* alongside the mean
preserves how much evidence varies across a clip, which a mean alone collapses.

**The head is small on purpose.** The predecessor documented a 4-layer d=512
encoder collapsing to a constant function on ~1,400 clips — logits agreeing to
six decimal places, loss pinned at ln(2). This corpus is 1,866.

**Augmentation happens before caching.** A frozen frontend cannot be augmented
on the fly, so training clips are encoded three times: clean plus two variants
perturbed with random EQ, bandwidth reduction, coloured noise and gain. All
target the *channel*, because the shortcut worth breaking is "recorded in a
quiet room and never re-encoded". SpecAugment-style masking on the cached
embeddings supplies the rest.

**Model selection watches val EER**, not loss or accuracy. Loss keeps improving
while the model sharpens scores it already ranks correctly; accuracy at a fixed
0.5 rewards whichever threshold the class balance happens to favour.

**It abstains on non-speech.** An energy gate refuses a verdict below 0.5s of
voiced audio. Silence is not evidence of synthesis; it is absence of evidence.
Through the API this surfaces as the uncertain band, never as a confident
"authentic".

**It scores long files in windows** — 8s at 4s hop, mean reported with max
alongside, so a short synthetic insert in a long genuine recording is not
averaged into invisibility. The predecessor analysed only the first 8 seconds
of any upload.

**Audio decoding has three tiers.** soundfile for the corpus, librosa for other
bare formats, ffmpeg for video containers — soundfile rejects containers
outright and librosa 1.0 dropped the audioread fallback, so without the third
tier every video upload fails at the front door.

## Tests

```bash
python -m pytest tests/ -q        # 10 passed
```

Covers filename→generator parsing, split leakage, EER correctness, padding
invariance of the head (padding must not change a score), and the silence case
that broke the predecessor.

## Layout

```
src/preprocessing/   manifest, splits, embeddings+cache, dataset, augment
src/models/head.py   attentive-pooling classifier
src/training/        train loop, metrics (EER/AUC/per-source)
src/evaluation/      temperature scaling, threshold, ECE
src/inference/       VAD gate, windowing, prediction
scripts/             pipeline steps + serve/router/start_all
outputs/run/         best.pt, policy.json, history.csv, test_results.json
```

Cache is ~1.3 GB (`data/cache/`), corpus ~570 MB (`data/raw/`). Neither is
meant for version control.

## Limitations

- **2.17 hours of training data.** Published anti-spoofing work uses hundreds.
- **14 source videos for the entire human class.** The corpus's weakest point.
- **78 calibration clips.** Temperature and threshold rest on very little.
- Expect worse than 2% EER against a modern voice-cloning system optimised to
  imitate a specific person.
- Nothing here detects *manipulated* human speech — only *synthetic* speech.
  Those are different problems, and the splicing case belongs to the visual
  branch or to `../deepfake-detect`.
