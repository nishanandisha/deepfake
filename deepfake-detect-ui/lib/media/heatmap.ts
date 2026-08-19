"use client";

import type { ScenarioId } from "@/lib/types";

/** Deterministic PRNG so a given sample+frame always renders the same overlay. */
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashSeed(str: string): number {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
  }
  return h;
}

/**
 * Synthetic Grad-CAM-style overlay: a canvas-drawn "jet" colormap blob,
 * transparent background, meant to be layered above an extracted video
 * frame. Not a real saliency map (no model runs client-side) — position
 * and intensity are driven by scenario + a per-frame seed so the demo is
 * visually coherent and reproducible, standing in for
 * src/explain/cam_visual.py's real Grad-CAM output.
 */
export function generateSaliencyOverlay(
  width: number,
  height: number,
  scenario: ScenarioId,
  seedKey: string,
  salienceScore: number
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const rand = mulberry32(hashSeed(seedKey));
  ctx.clearRect(0, 0, width, height);

  const isVisualImplicated = scenario === "fake_video" || scenario === "fake_both";
  const blobCount = isVisualImplicated ? 2 : 1;
  const baseIntensity = isVisualImplicated
    ? 0.35 + salienceScore * 0.5
    : 0.08 + salienceScore * 0.12;

  for (let b = 0; b < blobCount; b++) {
    // Face-plausible region: upper-center of frame.
    const cx = width * (0.42 + rand() * 0.16);
    const cy = height * (isVisualImplicated ? 0.32 + rand() * 0.18 : 0.3 + rand() * 0.4);
    const radius = Math.min(width, height) * (isVisualImplicated ? 0.24 + rand() * 0.1 : 0.3 + rand() * 0.15);

    const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    const alpha = Math.min(0.85, baseIntensity);
    gradient.addColorStop(0, `rgba(255, 64, 48, ${alpha})`);
    gradient.addColorStop(0.4, `rgba(255, 176, 0, ${alpha * 0.75})`);
    gradient.addColorStop(0.75, `rgba(80, 220, 255, ${alpha * 0.35})`);
    gradient.addColorStop(1, "rgba(80, 220, 255, 0)");

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
  }

  return canvas.toDataURL("image/png");
}
