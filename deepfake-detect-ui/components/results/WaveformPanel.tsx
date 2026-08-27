"use client";

import { motion } from "framer-motion";
import { AudioLines } from "lucide-react";
import type { TamperSegment, WaveformData } from "@/lib/types";
import { formatSpan } from "@/lib/analysis/localization";

/**
 * Shades only the regions that survived into confirmed audio segments, not
 * every raw `suspiciousRegions` entry the backend returns. Those are relative
 * scores that exist on clean clips too — painting them red would contradict
 * the "no manipulation found" verdict shown directly above.
 */
export function WaveformPanel({
  waveform,
  segments,
}: {
  waveform: WaveformData | null;
  segments: TamperSegment[];
}) {
  if (!waveform || waveform.peaks.length === 0) {
    return (
      <div className="glass-panel rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-foreground">Audio track</h3>
        <div className="mt-4 flex h-24 items-center justify-center gap-2 rounded-lg border border-dashed border-border text-xs text-muted-foreground">
          <AudioLines className="h-4 w-4" />
          No decodable audio track in this file.
        </div>
      </div>
    );
  }

  const { peaks, durationSeconds } = waveform;
  const audioSegments = segments.filter((s) => s.modality === "audio");

  return (
    <div className="glass-panel rounded-2xl p-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Audio track</h3>
        <span className="font-mono text-[11px] text-muted-foreground">
          {durationSeconds.toFixed(1)}s
        </span>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {audioSegments.length > 0
          ? "Red bands are the stretches flagged as synthetic speech — the same regions listed above."
          : "Nothing in this track resembled synthetic speech."}
      </p>

      <div className="relative mt-4 h-20 w-full">
        {audioSegments.map((segment) => (
          <div
            key={segment.id}
            title={formatSpan(segment)}
            className="absolute inset-y-0 rounded border-x border-deepfake/50 bg-deepfake"
            style={{
              left: `${(segment.startSeconds / durationSeconds) * 100}%`,
              width: `${((segment.endSeconds - segment.startSeconds) / durationSeconds) * 100}%`,
              opacity: 0.16 + segment.confidence * 0.24,
            }}
          />
        ))}
        <div className="absolute inset-0 flex items-center gap-px">
          {peaks.map((p, i) => (
            <motion.div
              key={i}
              initial={{ scaleY: 0 }}
              animate={{ scaleY: 1 }}
              transition={{ duration: 0.4, delay: Math.min(i * 0.003, 0.4) }}
              className="min-h-[2px] flex-1 rounded-full bg-gradient-to-t from-brand/70 to-brand-soft/70"
              style={{ height: `${Math.max(4, p * 100)}%` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
