# DeepFake — video & audio forensics UI

The front end for the multimodal detector in [`../deepfake-detect`](../deepfake-detect).

It answers two questions about an uploaded clip:

1. **Is this a deepfake?**
2. **If so, which parts of it were faked** — which seconds, in which track.

Built with Next.js 16 (App Router), Tailwind v4, zustand, framer-motion and
recharts.

## Running it

```bash
# 1. start the detectors (separate terminal)
bash ../ai-voice-detect/scripts/start_all.sh    # serves http://localhost:8000

# 2. start this UI
npm install
npm run dev              # serves http://localhost:3000
```

`NEXT_PUBLIC_INFERENCE_API` in `.env.local` points at the detector and
defaults to `http://localhost:8000`.

### What is behind port 8000

`start_all.sh` brings up a router plus both models:

```
this UI :3000 -> router :8000 -+-> :8001  ../deepfake-detect  (visual, advisory)
                               +-> :8002  ../ai-voice-detect  (WavLM audio, verdict)
```

The UI talks only to 8000 and needed no change to work with either. Three
processes exist because both model projects ship a top-level `src` package and
cannot be imported into one process.

The router probes each upload for a decodable video stream: audio-only files go
to the audio model alone, video files go to both and the scores merge as
`max(visual, audio)` — a clip is suspect if either branch implicates it. The
response carries `drivenBy`, naming the branch that set the score.

**The visual branch is advisory.** It scores blank frames at 0.0039 against a
real clip's 0.0038, while white/grey/noise frames read 0.84–0.98 "fake", and
its Haar face detector fails silently by centre-cropping background. Only the
audio branch has a validated figure — 2.00% EER on a held-out split including
two generators withheld from training.

To run the original multimodal backend alone instead, start
`../deepfake-detect/run_server.sh` on port 8000.

The API exposes `GET /api/health` and `POST /api/infer` (multipart, field
name `file`). Note it is **`/api/health`**, not `/health`.

### With the backend down

The UI falls back to a mock engine so it still demos. That state is never
implicit:

- the header badge switches from `live model` to `mock data`;
- a **Mock mode** scenario picker appears on the landing page, letting you
  choose what to simulate. It is hidden whenever the real model is up,
  because it decides nothing then.

## What the screen shows

| Panel | Source |
|---|---|
| Verdict + likelihood gauge | `cScore` against the calibrated `tauLo`/`tauHi` |
| What the model found | generated from this result's own numbers |
| Where the manipulation is | per-track timeline of implicated spans |
| Manipulated regions | expandable evidence + your per-region judgement |
| Your verdict | confirm or override the model |
| Modality attribution | fusion `gate` + per-branch probabilities |
| Acoustic feature attribution | KernelSHAP over named descriptors |
| Every sampled frame | full Grad-CAM filmstrip, implicated frames marked (video only) |
| Audio track | waveform with flagged regions shaded |
| How the call was made | calibrated thresholds and operating rates |

## Verdicts, not moderation actions

The backend returns a moderation-routing label — `approve` / `flag` / `block`
(see `src/evaluation/policy.py`). That is a *queue* decision and is kept
as-is on the wire, but nothing user-facing renders it. The UI maps it to the
question the product actually answers:

| Backend | UI verdict |
|---|---|
| `block` | **Deepfake** |
| `flag` | **Possibly manipulated** |
| `approve` | **Not a deepfake** |

The mapping is driven by `cScore` against the thresholds the backend ships
with each result, not by the `decision` string, so the boundary shown is the
boundary the model used. All of it lives in
[`lib/analysis/localization.ts`](lib/analysis/localization.ts).

Scores are also **flipped to read as manipulation likelihood** (`1 - cScore`).
The gauge and the verdict now point the same way — high means fake — instead
of the old authenticity score, which read backwards next to a red badge.

## Localizing the manipulation

Only part of a clip is usually altered, so a whole-clip score is not enough.
`buildTamperSegments()` turns evidence the backend already returns into
timestamped spans:

- **Video** — frames whose Grad-CAM salience clears `SALIENCE_FLOOR` (0.5),
  with consecutive flagged frames merged into one span and padded by half a
  sampling interval on each side.
- **Audio** — `waveform.suspiciousRegions`, merged when less than 0.3 s
  apart.

Two rules keep this honest:

- **A branch must implicate its own modality first.** Salience and acoustic
  suspicion are *relative* — every clip has a most-salient frame, authentic
  ones included. Unless a branch scores at least `BRANCH_IMPLICATION_FLOOR`
  (0.5) fake, none of its regions are surfaced. Without this the UI would
  invent "faked at 0:03" on a clean clip.
- **Segment confidence is scaled by that branch probability**, so a region's
  displayed confidence can never exceed the branch's own conviction.

If a branch implicates its modality but no single frame clears the floor, the
strongest frame is surfaced anyway — the evidence is spread thin rather than
absent, and the reviewer still gets somewhere to look.

Timestamps render to tenths (`0:03.4–0:03.7`). Flagged audio regions are
routinely under a second, and whole-second rounding collapsed them into
`0:03–0:03`.

## Audio-only files (voice notes)

MP3/WAV/FLAC uploads with no video track are fully supported. The router
probes for a decodable video stream and, finding none, sends the file to
[`../ai-voice-detect`](../ai-voice-detect) alone — a frozen WavLM-Base+
frontend with a trained attentive-pooling head, measured at **2.00% EER** on a
held-out split that includes two TTS generators withheld from training.

Two behaviours differ from the old acoustic branch, both deliberate:

- **Non-speech abstains.** Below 0.5s of voiced audio the model returns no
  verdict and the response lands in the uncertain band. The predecessor scored
  ten seconds of digital silence at 0.98 "fake"; absence of evidence is not
  evidence of synthesis.
- **Long files are scored in windows** — 8s at 4s hop, aggregated — so a short
  synthetic insert in a long genuine recording is not averaged away. The
  predecessor analysed only the first 8 seconds of any upload.

Note the narrower question: this model detects **synthetic speech**, not
*manipulated* speech. Spliced real human audio is genuinely human by its
reckoning, and that case belongs to the visual branch.

The response sets `hasVideo: false`, and the UI adapts throughout:

- the video timeline row reads *"No video track in this file"* rather than
  *"no manipulation detected"* — an absent track is not a clean one;
- Modality attribution drops the cross-modal gate and shows the acoustic
  branch alone;
- the Grad-CAM filmstrip is hidden entirely;
- the SHAP chart is empty. `ai-voice-detect` uses learned WavLM embeddings
  rather than 68 named descriptors, so there are no human-readable feature
  names to attribute to. It returns `acousticShap: []`; inventing labels
  would be worse than showing none.

Localization still works — per-window scores become suspicious regions, so
audio is timestamped exactly as it is for a video upload.

> Calibration is no longer approximated. The old audio-only path reused the
> temperature and thresholds fitted on the **fused** logit, which capped
> synthetic speech at `flag` and never reached `block`. `ai-voice-detect`
> fits its own temperature (1.19) and threshold (0.106) on a dedicated
> audio-only calibration split.

> The older `../audio-deepfake-detect` package (test AUC 0.972, MFCC features)
> is superseded and serves nothing. Its representation was the accuracy
> ceiling: mel-binning plus DCT truncation discards the phase and fine
> spectral structure where TTS artifacts live, which is why it scored silence
> at 0.98 "fake".

## Reviewing the model

Every region can be expanded to its evidence — Grad-CAM frames for video,
the SHAP descriptors that drove it for audio — and marked **Yes, manipulated**
or **No, looks genuine**. A running tally feeds the final *Your verdict*
panel, where the model's call can be confirmed or overridden. This matters
when only a few seconds of a clip are altered and the whole-clip score
overstates or understates the case.

## Color system

Four source colors, used for every surface, text tone, accent and chart
series:

| Hex | RGB | Token | Role |
|---|---|---|---|
| `#1B262C` | `27, 38, 44` | `--background` | page ground |
| `#0F4C75` | `15, 76, 117` | `--brand-deep` | raised surfaces, fills |
| `#3282B8` | `50, 130, 184` | `--brand` | primary accent, visual stream |
| `#BBE1FA` | `187, 225, 250` | `--brand-soft` / `--foreground` | text, acoustic stream |

Three status hues sit outside the palette **on purpose**, and are used only
for verdict signalling — teal `#3FBFA0` (authentic), amber `#E8B04B`
(uncertain), red `#E5484D` (deepfake). The palette is monochromatic blue, so
a verdict rendered in it alone could not separate "deepfake" from "not
deepfake" at a glance, which is the one thing this screen exists to say.

Tokens are defined once in [`app/globals.css`](app/globals.css); components
reference `--brand`, `--brand-soft`, `--brand-deep`, `--authentic`,
`--uncertain` and `--deepfake` rather than raw hex.

## Development

```bash
npx tsc --noEmit     # typecheck
npx eslint .         # lint
npm run build        # production build
```
