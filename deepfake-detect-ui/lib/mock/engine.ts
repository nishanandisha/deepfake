"use client";

import { extractFrames } from "@/lib/media/extractFrames";
import { generateSaliencyOverlay } from "@/lib/media/heatmap";
import { decodeWaveform } from "@/lib/media/waveform";
import {
  FALSE_SUPPRESSION_RATE,
  REVIEW_QUEUE_RATE,
  SCENARIOS,
  TAU_HI,
  TAU_LO,
  decisionForScore,
} from "@/lib/mock/scenarios";
import type { InferenceResult, SaliencyFrame, ScenarioId, WaveformData } from "@/lib/types";

const FRAME_COUNT = 6;

function jitter(base: number, spread: number): number {
  const value = base + (Math.random() * 2 - 1) * spread;
  return Math.min(0.99, Math.max(0.01, value));
}

function makeSampleId(): string {
  return `sample_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

async function buildFallbackFrames(scenario: ScenarioId, count: number): Promise<
  { frameIndex: number; timestamp: number; dataUrl: string; width: number; height: number }[]
> {
  const width = 480;
  const height = 270;
  const frames = [];
  for (let i = 0; i < count; i++) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, "#161c2c");
      gradient.addColorStop(1, "#0a0d15");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      for (let b = 0; b < 24; b++) {
        const bw = 4;
        const bh = 20 + Math.sin(i * 12 + b) * 40 + 60;
        ctx.fillRect(b * (width / 24), height / 2 - bh / 2, bw, bh);
      }
    }
    frames.push({
      frameIndex: i,
      timestamp: i,
      dataUrl: canvas.toDataURL("image/jpeg", 0.85),
      width,
      height,
    });
  }
  return frames;
}

export interface RunOptions {
  file: File;
  scenario: ScenarioId;
  onStage?: (stageIndex: number) => void;
}

/**
 * Simulates the full inference pipeline: real frame extraction + waveform
 * decode from the uploaded file, combined with scenario-driven mock scoring
 * shaped to match the real backend's InferenceResult contract.
 */
export async function runMockInference({ file, scenario, onStage }: RunOptions): Promise<InferenceResult> {
  const def = SCENARIOS[scenario];
  const sampleId = makeSampleId();

  onStage?.(0); // preprocessing
  const isVideo = file.type.startsWith("video/");

  let rawFrames;
  try {
    rawFrames = isVideo ? await extractFrames(file, FRAME_COUNT) : await buildFallbackFrames(scenario, FRAME_COUNT);
  } catch {
    rawFrames = await buildFallbackFrames(scenario, FRAME_COUNT);
  }

  let waveform: WaveformData | null = null;
  try {
    const decoded = await decodeWaveform(file);
    const suspicious =
      scenario === "fake_audio" || scenario === "fake_both"
        ? [
            {
              startSeconds: decoded.durationSeconds * 0.15,
              endSeconds: decoded.durationSeconds * 0.4,
              intensity: 0.75,
            },
            {
              startSeconds: decoded.durationSeconds * 0.6,
              endSeconds: decoded.durationSeconds * 0.82,
              intensity: 0.55,
            },
          ]
        : [];
    waveform = { peaks: decoded.peaks, durationSeconds: decoded.durationSeconds, suspiciousRegions: suspicious };
  } catch {
    waveform = null;
  }

  onStage?.(1); // visual branch
  await delay(280);

  onStage?.(2); // acoustic branch
  await delay(280);

  onStage?.(3); // fusion
  await delay(320);

  const cScore = jitter(def.cScoreBase, def.cScoreJitter);
  const gate = jitter(def.gateBase, def.gateJitter);
  const yHatVisual = jitter(def.yHatVisualBase, 0.04);
  const yHatAcoustic = jitter(def.yHatAcousticBase, 0.04);
  const yHatFused = 1 - cScore;

  onStage?.(4); // calibration
  await delay(240);
  const decision = decisionForScore(cScore);

  onStage?.(5); // explanation
  await delay(320);

  const isVisualImplicated = scenario === "fake_video" || scenario === "fake_both";
  const salienceBase = isVisualImplicated ? 0.55 : 0.15;

  const visualSaliency: SaliencyFrame[] = rawFrames.map((f, i) => {
    const salienceScore = Math.min(1, Math.max(0, salienceBase + (i % 3 === 1 ? 0.25 : 0) + Math.random() * 0.15));
    const heatmapUrl = generateSaliencyOverlay(
      f.width,
      f.height,
      scenario,
      `${sampleId}_${f.frameIndex}`,
      salienceScore
    );
    return {
      frameIndex: f.frameIndex,
      timestamp: f.timestamp,
      salienceScore,
      thumbnailUrl: f.dataUrl,
      heatmapUrl,
    };
  });

  return {
    sampleId,
    fileName: file.name || "uploaded_clip",
    createdAt: new Date().toISOString(),
    decision,
    cScore,
    tauLo: TAU_LO,
    tauHi: TAU_HI,
    gate,
    yHatVisual,
    hasVideo: true,
    yHatAcoustic,
    yHatFused,
    acousticShap: def.acousticShap.map((entry) => ({
      feature: entry.feature,
      value: entry.value * (0.85 + Math.random() * 0.3),
    })),
    visualSaliency,
    waveform,
    falseSuppressionRate: FALSE_SUPPRESSION_RATE,
    reviewQueueRate: REVIEW_QUEUE_RATE,
    manipulatedModalityGuess: def.manipulatedModalityGuess,
    scenario,
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
