"use client";

export interface DecodedWaveform {
  peaks: number[];
  durationSeconds: number;
}

/**
 * Real, client-side audio decode via the Web Audio API. Works for both
 * plain audio files and the audio track embedded in a video file.
 */
export async function decodeWaveform(file: File, bucketCount = 240): Promise<DecodedWaveform> {
  const arrayBuffer = await file.arrayBuffer();
  const AudioContextCtor =
    window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const audioCtx = new AudioContextCtor();

  try {
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    const channelData = audioBuffer.getChannelData(0);
    const bucketSize = Math.max(1, Math.floor(channelData.length / bucketCount));
    const peaks: number[] = [];

    for (let i = 0; i < bucketCount; i++) {
      const start = i * bucketSize;
      const end = Math.min(start + bucketSize, channelData.length);
      let max = 0;
      for (let j = start; j < end; j++) {
        const abs = Math.abs(channelData[j]);
        if (abs > max) max = abs;
      }
      peaks.push(max);
    }

    const maxPeak = Math.max(...peaks, 1e-6);
    return {
      peaks: peaks.map((p) => p / maxPeak),
      durationSeconds: audioBuffer.duration,
    };
  } finally {
    await audioCtx.close();
  }
}
