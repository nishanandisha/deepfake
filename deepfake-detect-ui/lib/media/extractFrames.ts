"use client";

export interface ExtractedFrame {
  frameIndex: number;
  timestamp: number;
  dataUrl: string;
  width: number;
  height: number;
}

/**
 * Real, client-side frame extraction from an uploaded video file via
 * an offscreen <video> element + <canvas>. Used so the moderator UI shows
 * actual content from the upload rather than stock imagery.
 */
export async function extractFrames(
  file: File,
  count = 6,
  maxDimension = 480
): Promise<ExtractedFrame[]> {
  const objectUrl = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = objectUrl;

  try {
    await waitFor(video, "loadedmetadata");

    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 4;
    const scale = Math.min(1, maxDimension / Math.max(video.videoWidth || maxDimension, 1));
    const width = Math.max(1, Math.round((video.videoWidth || maxDimension) * scale));
    const height = Math.max(1, Math.round((video.videoHeight || maxDimension) * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");

    const frames: ExtractedFrame[] = [];
    const padding = duration * 0.06;
    for (let i = 0; i < count; i++) {
      const t = padding + (i / Math.max(count - 1, 1)) * (duration - padding * 2);
      const timestamp = clamp(t, 0, Math.max(duration - 0.05, 0));
      await seekTo(video, timestamp);
      ctx.drawImage(video, 0, 0, width, height);
      frames.push({
        frameIndex: i,
        timestamp,
        dataUrl: canvas.toDataURL("image/jpeg", 0.85),
        width,
        height,
      });
    }
    return frames;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function waitFor(el: HTMLVideoElement, event: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const onError = () => {
      el.removeEventListener(event, onLoad);
      reject(new Error(`Video failed to load (${event})`));
    };
    const onLoad = () => {
      el.removeEventListener("error", onError);
      resolve();
    };
    el.addEventListener(event, onLoad, { once: true });
    el.addEventListener("error", onError, { once: true });
  });
}

function seekTo(video: HTMLVideoElement, timestamp: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const onSeeked = () => {
      video.removeEventListener("error", onError);
      resolve();
    };
    const onError = () => {
      video.removeEventListener("seeked", onSeeked);
      reject(new Error("Video seek failed"));
    };
    video.addEventListener("seeked", onSeeked, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.currentTime = timestamp;
  });
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
