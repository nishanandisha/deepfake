"use client";

import { motion } from "framer-motion";
import { Ear, Eye } from "lucide-react";
import type { SegmentModality, TamperSegment } from "@/lib/types";
import { formatPreciseTimecode, formatTimecode } from "@/lib/analysis/localization";
import { cn } from "@/lib/utils";

const TRACKS: { modality: SegmentModality; label: string; icon: typeof Eye }[] = [
  { modality: "video", label: "Video", icon: Eye },
  { modality: "audio", label: "Audio", icon: Ear },
];

function Track({
  modality,
  label,
  icon: Icon,
  segments,
  duration,
  selectedId,
  onSelect,
  absent,
}: {
  modality: SegmentModality;
  label: string;
  icon: typeof Eye;
  segments: TamperSegment[];
  duration: number;
  selectedId: string | null;
  onSelect: (segment: TamperSegment) => void;
  absent?: boolean;
}) {
  const mine = segments.filter((s) => s.modality === modality);

  return (
    <div className="flex items-center gap-3">
      <span className="flex w-16 shrink-0 items-center gap-1.5 text-xs font-medium text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </span>

      <div
        className={cn(
          "relative h-9 flex-1 overflow-hidden rounded-lg border border-border",
          absent ? "border-dashed bg-secondary/20" : "bg-secondary/60"
        )}
      >
        {mine.length === 0 ? (
          <span className="absolute inset-0 flex items-center justify-center text-[11px] text-muted-foreground">
            {absent
              ? "No video track in this file"
              : "No manipulation detected in this track"}
          </span>
        ) : (
          mine.map((segment) => {
            const left = (segment.startSeconds / duration) * 100;
            const width = ((segment.endSeconds - segment.startSeconds) / duration) * 100;
            const active = segment.id === selectedId;
            return (
              <motion.button
                key={segment.id}
                type="button"
                onClick={() => onSelect(segment)}
                title={`${formatPreciseTimecode(segment.startSeconds)}–${formatPreciseTimecode(segment.endSeconds)} · ${Math.round(segment.confidence * 100)}% confidence`}
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                style={{
                  left: `${left}%`,
                  width: `${Math.max(width, 1.5)}%`,
                  opacity: 0.45 + segment.confidence * 0.55,
                  transformOrigin: "left",
                }}
                className={cn(
                  "absolute inset-y-0 rounded-[3px] bg-deepfake transition-shadow",
                  active && "ring-2 ring-brand-soft ring-offset-1 ring-offset-background"
                )}
              />
            );
          })
        )}
      </div>
    </div>
  );
}

export function TamperTimeline({
  segments,
  duration,
  selectedId,
  onSelect,
  audioOnly = false,
}: {
  segments: TamperSegment[];
  duration: number;
  selectedId: string | null;
  onSelect: (segment: TamperSegment) => void;
  audioOnly?: boolean;
}) {
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="glass-panel rounded-2xl p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-foreground">
            {segments.length > 0 ? "Where the manipulation is" : "Timeline"}
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            {segments.length > 0
              ? "Each red block is a stretch of the clip the model implicates. Select one to inspect the evidence behind it."
              : audioOnly
                ? "The speech was checked end to end and was not implicated at any point."
                : "Both tracks were checked end to end and neither was implicated at any point."}
          </p>
        </div>
        <span className="shrink-0 rounded-full border border-border px-2.5 py-1 font-mono text-[11px] text-muted-foreground">
          {duration.toFixed(1)}s
        </span>
      </div>

      <div className="mt-5 space-y-2.5">
        {TRACKS.map((track) => (
          <Track
            key={track.modality}
            {...track}
            segments={segments}
            duration={duration}
            selectedId={selectedId}
            onSelect={onSelect}
            absent={audioOnly && track.modality === "video"}
          />
        ))}
      </div>

      <div className="ml-[76px] mt-2 flex justify-between font-mono text-[10px] text-muted-foreground">
        {ticks.map((t) => (
          <span key={t}>{formatTimecode(t * duration)}</span>
        ))}
      </div>

      <div className={cn(
        "mt-4 flex items-center gap-4 border-t border-border pt-3 text-[11px] text-muted-foreground",
        segments.length === 0 && "hidden"
      )}>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-4 rounded-[2px] bg-secondary" />
          clean
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-4 rounded-[2px] bg-deepfake/60" />
          implicated
        </span>
        <span className="ml-auto hidden sm:inline">opacity scales with confidence</span>
      </div>
    </div>
  );
}
