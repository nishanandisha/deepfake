import { friendlyName } from "@/lib/analysis/featureNames";
import type {
  InferenceResult,
  SaliencyFrame,
  SegmentModality,
  TamperSegment,
  Verdict,
} from "@/lib/types";

/* -------------------------------------------------------------------------
   Verdict
   ------------------------------------------------------------------------- */

/**
 * Collapses the backend's three-way routing decision into the question the
 * product actually answers.
 *
 * Driven by the calibrated thresholds the backend ships with the result
 * (tauLo/tauHi from src/evaluation/policy.py) rather than the `decision`
 * string, so the boundary shown in the UI is the boundary the model used.
 */
export function verdictFor(result: InferenceResult): Verdict {
  if (result.cScore < result.tauLo) return "deepfake";
  if (result.cScore >= result.tauHi) return "authentic";
  return "uncertain";
}

/** 0..1 likelihood the clip is manipulated. cScore is authenticity, so invert. */
export function deepfakeScore(result: InferenceResult): number {
  return 1 - result.cScore;
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  deepfake: "Deepfake",
  uncertain: "Possibly manipulated",
  authentic: "Not a deepfake",
};

export const VERDICT_BLURB: Record<Verdict, string> = {
  deepfake: "The model is confident this clip has been manipulated.",
  uncertain:
    "This clip falls between the calibrated thresholds — the model cannot decide on its own and needs your review.",
  authentic: "No manipulation found in either the video or the audio track.",
};

/* -------------------------------------------------------------------------
   Localization
   ------------------------------------------------------------------------- */

/**
 * A branch has to actually implicate its modality before we point at
 * timestamps within it. Grad-CAM salience and acoustic suspicion scores are
 * *relative* — every clip has a most-salient frame, including authentic ones.
 * Surfacing "faked at 0:03" off a branch that scored 0.02 fake would invent a
 * finding the model never made.
 */
const BRANCH_IMPLICATION_FLOOR = 0.5;

/** Frames at or above this salience are treated as implicated. */
const SALIENCE_FLOOR = 0.5;

/** Audio regions closer together than this are one event, not two. */
const AUDIO_MERGE_GAP_SECONDS = 0.3;

/** Nothing shorter than this is legible on a timeline, so pad around the peak. */
const MIN_SEGMENT_SECONDS = 0.35;

export function clipDuration(result: InferenceResult): number {
  if (result.waveform?.durationSeconds) return result.waveform.durationSeconds;
  const last = result.visualSaliency.at(-1);
  return last ? last.timestamp + 1 : 1;
}

/** Audio-only uploads get an acoustic-only pass; the UI must not imply a picture. */
export function isAudioOnly(result: InferenceResult): boolean {
  return result.hasVideo === false;
}

function medianGap(frames: SaliencyFrame[]): number {
  if (frames.length < 2) return 1;
  const gaps = frames.slice(1).map((f, i) => f.timestamp - frames[i].timestamp);
  const sorted = [...gaps].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] || 1;
}

function padded(start: number, end: number, duration: number): [number, number] {
  if (end - start >= MIN_SEGMENT_SECONDS) {
    return [Math.max(0, start), Math.min(duration, end)];
  }
  const mid = (start + end) / 2;
  const half = MIN_SEGMENT_SECONDS / 2;
  return [Math.max(0, mid - half), Math.min(duration, mid + half)];
}

function videoSegments(result: InferenceResult, duration: number): TamperSegment[] {
  const branchProb = result.yHatVisual;
  if (branchProb === null || branchProb < BRANCH_IMPLICATION_FLOOR) return [];

  const frames = [...result.visualSaliency].sort((a, b) => a.timestamp - b.timestamp);
  if (frames.length === 0) return [];

  let flaggedIdx = frames
    .map((f, i) => ({ f, i }))
    .filter(({ f }) => f.salienceScore >= SALIENCE_FLOOR)
    .map(({ i }) => i);

  // The branch says "fake" but no single frame cleared the bar — the evidence
  // is spread thin rather than absent. Fall back to the strongest frame so the
  // reviewer still gets a place to look.
  if (flaggedIdx.length === 0) {
    const best = frames.reduce((a, b) => (b.salienceScore > a.salienceScore ? b : a));
    flaggedIdx = [frames.indexOf(best)];
  }

  const gap = medianGap(frames);
  const runs: number[][] = [];
  for (const idx of flaggedIdx) {
    const tail = runs.at(-1);
    if (tail && idx === tail.at(-1)! + 1) tail.push(idx);
    else runs.push([idx]);
  }

  return runs.map((run, n) => {
    const runFrames = run.map((i) => frames[i]);
    const peak = runFrames.reduce((a, b) => (b.salienceScore > a.salienceScore ? b : a));
    const [startSeconds, endSeconds] = padded(
      runFrames[0].timestamp - gap / 2,
      runFrames.at(-1)!.timestamp + gap / 2,
      duration
    );
    return {
      id: `video_${n}`,
      modality: "video" as SegmentModality,
      startSeconds,
      endSeconds,
      peakSeconds: peak.timestamp,
      confidence: peak.salienceScore * branchProb,
      frames: runFrames,
      topFeatures: [],
    };
  });
}

function audioSegments(result: InferenceResult, duration: number): TamperSegment[] {
  const branchProb = result.yHatAcoustic;
  if (branchProb < BRANCH_IMPLICATION_FLOOR) return [];

  const regions = [...(result.waveform?.suspiciousRegions ?? [])].sort(
    (a, b) => a.startSeconds - b.startSeconds
  );
  if (regions.length === 0) return [];

  const topFeatures = [...result.acousticShap]
    .filter((e) => e.value > 0) // positive SHAP pushes toward "fake"
    .sort((a, b) => b.value - a.value)
    .slice(0, 3)
    .map((e) => friendlyName(e.feature));

  type Merged = { start: number; end: number; peak: number; intensity: number };
  const merged: Merged[] = [];
  for (const r of regions) {
    const tail = merged.at(-1);
    if (tail && r.startSeconds - tail.end <= AUDIO_MERGE_GAP_SECONDS) {
      tail.end = Math.max(tail.end, r.endSeconds);
      if (r.intensity > tail.intensity) {
        tail.intensity = r.intensity;
        tail.peak = (r.startSeconds + r.endSeconds) / 2;
      }
    } else {
      merged.push({
        start: r.startSeconds,
        end: r.endSeconds,
        peak: (r.startSeconds + r.endSeconds) / 2,
        intensity: r.intensity,
      });
    }
  }

  return merged.map((m, n) => {
    const [startSeconds, endSeconds] = padded(m.start, m.end, duration);
    return {
      id: `audio_${n}`,
      modality: "audio" as SegmentModality,
      startSeconds,
      endSeconds,
      peakSeconds: m.peak,
      confidence: m.intensity * branchProb,
      frames: [],
      topFeatures,
    };
  });
}

/**
 * Every stretch of the clip the model implicates, in both modalities,
 * strongest first.
 */
export function buildTamperSegments(result: InferenceResult): TamperSegment[] {
  const duration = clipDuration(result);
  return [...videoSegments(result, duration), ...audioSegments(result, duration)].sort(
    (a, b) => b.confidence - a.confidence
  );
}

/* -------------------------------------------------------------------------
   Formatting + narration
   ------------------------------------------------------------------------- */

export function formatTimecode(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Tenths included. Manipulated regions are routinely shorter than a second,
 * and whole-second rounding collapsed those into "0:03-0:03".
 */
export function formatPreciseTimecode(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

export function formatSpan(segment: TamperSegment): string {
  return `${formatPreciseTimecode(segment.startSeconds)}–${formatPreciseTimecode(segment.endSeconds)}`;
}

/** Total implicated time per modality, for the summary line. */
export function tamperedDuration(segments: TamperSegment[], modality: SegmentModality): number {
  return segments
    .filter((s) => s.modality === modality)
    .reduce((total, s) => total + (s.endSeconds - s.startSeconds), 0);
}

/**
 * Builds the summary from this result's own numbers.
 *
 * Replaces the previous canned per-scenario paragraph, which was keyed off
 * the scenario picker and so could describe a clip the model had scored
 * differently.
 */
export function buildSummary(result: InferenceResult, segments: TamperSegment[]): string {
  const verdict = verdictFor(result);
  const pct = Math.round(deepfakeScore(result) * 100);
  const duration = clipDuration(result);

  if (verdict === "authentic" || segments.length === 0) {
    if (verdict === "authentic") {
      return result.hasVideo
        ? `Both the visual and acoustic branches read this clip as genuine capture (${pct}% manipulation likelihood). No frame span or audio region crossed the threshold for manipulation, so there is nothing to localize.`
        : `This is an audio-only file, so only the acoustic branch ran. It reads the speech as genuine (${pct}% manipulation likelihood) and no region crossed the threshold for synthetic speech.`;
    }
    return `The fused score puts this clip at ${pct}% manipulation likelihood, but neither branch implicated its own modality strongly enough to point at specific timestamps. Treat the score as a whole-clip signal and review the full recording.`;
  }

  const videoSegs = segments.filter((s) => s.modality === "video");
  const audioSegs = segments.filter((s) => s.modality === "audio");

  const parts: string[] = [];
  if (videoSegs.length) {
    const secs = tamperedDuration(segments, "video");
    parts.push(
      `${videoSegs.length} video ${videoSegs.length === 1 ? "span" : "spans"} (${secs.toFixed(1)}s of ${duration.toFixed(1)}s), strongest at ${formatTimecode(videoSegs[0].peakSeconds)}`
    );
  }
  if (audioSegs.length) {
    const secs = tamperedDuration(segments, "audio");
    parts.push(
      `${audioSegs.length} audio ${audioSegs.length === 1 ? "region" : "regions"} (${secs.toFixed(1)}s of ${duration.toFixed(1)}s), strongest at ${formatTimecode(audioSegs[0].peakSeconds)}`
    );
  }

  const noun = result.hasVideo ? "clip" : "recording";
  const lead =
    verdict === "deepfake"
      ? `This ${noun} is ${pct}% likely to be manipulated.`
      : `This ${noun} sits in the uncertain band at ${pct}% manipulation likelihood.`;

  const where = `The evidence is concentrated in ${parts.join(" and ")}.`;

  const scope = !result.hasVideo
    ? "There is no video track in this file, so only the speech was examined."
    : videoSegs.length && audioSegs.length
      ? "Both tracks are implicated."
      : videoSegs.length
        ? "The audio track shows no manipulation — only the picture is implicated."
        : "The picture shows no manipulation — only the audio is implicated.";

  return `${lead} ${where} ${scope} Step through each region below and confirm or reject it.`;
}
