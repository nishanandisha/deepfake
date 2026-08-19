"use client";

import { motion } from "framer-motion";
import { AudioLines } from "lucide-react";
import type { WaveformData } from "@/lib/types";

export function WaveformPanel({ waveform }: { waveform: WaveformData | null }) {
  if (!waveform || waveform.peaks.length === 0) {
    return (
      <div className="glass-panel rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-foreground">Audio waveform</h3>
        <div className="mt-4 flex h-24 items-center justify-center gap-2 rounded-lg border border-dashed border-white/10 text-xs text-muted-foreground">
          <AudioLines className="h-4 w-4" />
          No decodable audio track in this file.
        </div>
      </div>
    );
  }

  const { peaks, durationSeconds, suspiciousRegions } = waveform;

  return (
    <div className="glass-panel rounded-2xl p-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Audio waveform</h3>
        <span className="font-mono text-[11px] text-muted-foreground">
          {durationSeconds.toFixed(1)}s
        </span>
      </div>
      {suspiciousRegions.length > 0 && (
        <p className="mt-1 text-xs text-muted-foreground">
          Shaded bands mark segments where acoustic descriptors most resembled synthetic speech.
        </p>
      )}

      <div className="relative mt-4 h-20 w-full">
        {suspiciousRegions.map((region, i) => (
          <div
            key={i}
            className="absolute inset-y-0 rounded border-x border-block/40 bg-block"
            style={{
              left: `${(region.startSeconds / durationSeconds) * 100}%`,
              width: `${((region.endSeconds - region.startSeconds) / durationSeconds) * 100}%`,
              opacity: 0.16 + region.intensity * 0.22,
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
              className="min-h-[2px] flex-1 rounded-full bg-gradient-to-t from-cyan/70 to-violet/70"
              style={{ height: `${Math.max(4, p * 100)}%` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
