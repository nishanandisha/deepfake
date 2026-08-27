"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown, Ear, Eye, X } from "lucide-react";
import type { SegmentJudgement, TamperSegment } from "@/lib/types";
import { formatPreciseTimecode, formatSpan, formatTimecode } from "@/lib/analysis/localization";
import { cn } from "@/lib/utils";

function VideoEvidence({ segment }: { segment: TamperSegment }) {
  const [overlayOn, setOverlayOn] = useState(true);

  if (segment.frames.length === 0) return null;

  return (
    <div>
      <div className="flex items-center justify-between">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
          Frames in this span
        </p>
        <button
          type="button"
          onClick={() => setOverlayOn((v) => !v)}
          className={cn(
            "rounded-md border px-2 py-1 text-[11px] font-medium transition",
            overlayOn
              ? "border-brand/50 bg-brand/10 text-brand"
              : "border-border text-muted-foreground hover:text-foreground"
          )}
        >
          {overlayOn ? "Hide Grad-CAM" : "Show Grad-CAM"}
        </button>
      </div>
      <div className="mt-2.5 grid grid-cols-3 gap-2 sm:grid-cols-4">
        {segment.frames.map((frame) => (
          <figure
            key={frame.frameIndex}
            className="relative aspect-video overflow-hidden rounded-lg border border-border"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={frame.thumbnailUrl} alt="" className="h-full w-full object-cover" />
            {overlayOn && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={frame.heatmapUrl}
                alt=""
                className="absolute inset-0 h-full w-full object-cover"
              />
            )}
            <figcaption className="absolute bottom-1 right-1 rounded bg-background/80 px-1 font-mono text-[9px] text-foreground">
              {formatTimecode(frame.timestamp)}
            </figcaption>
          </figure>
        ))}
      </div>
      <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
        Brighter regions are where the visual branch found the strongest manipulation artefacts —
        typically blending seams around the jaw, mouth and hairline.
      </p>
    </div>
  );
}

function AudioEvidence({ segment }: { segment: TamperSegment }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
        Acoustic descriptors driving this region
      </p>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {segment.topFeatures.length > 0 ? (
          segment.topFeatures.map((f) => (
            <span
              key={f}
              className="rounded-md border border-border bg-secondary/50 px-2 py-1 font-mono text-[11px] text-foreground"
            >
              {f}
            </span>
          ))
        ) : (
          <span className="text-[11px] text-muted-foreground">
            No individual descriptor dominated this region.
          </span>
        )}
      </div>
      <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
        These descriptors departed furthest from natural speech across{" "}
        {formatSpan(segment)} — the signature of vocoded or cloned audio.
      </p>
    </div>
  );
}

function SegmentRow({
  segment,
  index,
  expanded,
  judgement,
  onToggle,
  onJudge,
}: {
  segment: TamperSegment;
  index: number;
  expanded: boolean;
  judgement: SegmentJudgement | undefined;
  onToggle: () => void;
  onJudge: (j: SegmentJudgement) => void;
}) {
  const Icon = segment.modality === "video" ? Eye : Ear;

  return (
    <li
      className={cn(
        "overflow-hidden rounded-xl border transition-colors",
        expanded ? "border-brand/40 bg-brand/[0.04]" : "border-border bg-secondary/25"
      )}
    >
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-3 px-4 py-3 text-left"
      >
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-deepfake/15 text-deepfake">
          <Icon className="h-3.5 w-3.5" />
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex items-baseline gap-2">
            <span className="font-mono text-sm font-medium text-foreground">
              {formatSpan(segment)}
            </span>
            <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
              {segment.modality}
            </span>
          </span>
          <span className="mt-0.5 block text-[11px] text-muted-foreground">
            Region {index + 1} · peak at {formatPreciseTimecode(segment.peakSeconds)} ·{" "}
            {Math.round(segment.confidence * 100)}% confidence
          </span>
        </span>

        {judgement && (
          <span
            className={cn(
              "shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium",
              judgement === "agree"
                ? "border-deepfake/40 bg-deepfake/10 text-deepfake"
                : "border-authentic/40 bg-authentic/10 text-authentic"
            )}
          >
            {judgement === "agree" ? "confirmed" : "rejected"}
          </span>
        )}

        <span className="hidden h-1.5 w-24 shrink-0 overflow-hidden rounded-full bg-secondary sm:block">
          <span
            className="block h-full bg-deepfake/70"
            style={{ width: `${Math.round(segment.confidence * 100)}%` }}
          />
        </span>

        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-muted-foreground transition-transform",
            expanded && "rotate-180"
          )}
        />
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="space-y-4 border-t border-border px-4 py-4">
              {segment.modality === "video" ? (
                <VideoEvidence segment={segment} />
              ) : (
                <AudioEvidence segment={segment} />
              )}

              <div className="flex flex-col gap-2 border-t border-border pt-3 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-[11px] text-muted-foreground">
                  Does this region really look manipulated to you?
                </p>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => onJudge("agree")}
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition",
                      judgement === "agree"
                        ? "border-deepfake/60 bg-deepfake/15 text-deepfake"
                        : "border-border text-muted-foreground hover:border-deepfake/50 hover:text-deepfake"
                    )}
                  >
                    <Check className="h-3.5 w-3.5" />
                    Yes, manipulated
                  </button>
                  <button
                    type="button"
                    onClick={() => onJudge("disagree")}
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition",
                      judgement === "disagree"
                        ? "border-authentic/60 bg-authentic/15 text-authentic"
                        : "border-border text-muted-foreground hover:border-authentic/50 hover:text-authentic"
                    )}
                  >
                    <X className="h-3.5 w-3.5" />
                    No, looks genuine
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  );
}

export function SegmentReview({
  segments,
  selectedId,
  onSelect,
  judgements,
  onJudge,
}: {
  segments: TamperSegment[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  judgements: Record<string, SegmentJudgement>;
  onJudge: (id: string, judgement: SegmentJudgement) => void;
}) {
  const reviewed = segments.filter((s) => judgements[s.id]).length;

  if (segments.length === 0) {
    return (
      <div className="glass-panel rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-foreground">Manipulated regions</h3>
        <div className="mt-4 flex items-center gap-2.5 rounded-xl border border-dashed border-authentic/30 bg-authentic/[0.04] px-4 py-6 text-xs text-muted-foreground">
          <Check className="h-4 w-4 shrink-0 text-authentic" />
          Neither branch implicated its own modality, so there is no region to localize. The clip
          reads as genuine end to end.
        </div>
      </div>
    );
  }

  return (
    <div className="glass-panel rounded-2xl p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-foreground">
            Manipulated regions
            <span className="ml-2 font-mono text-xs font-normal text-muted-foreground">
              {segments.length}
            </span>
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Only part of a clip is usually altered. Open each region, look at the evidence, and
            record whether you agree.
          </p>
        </div>
        <span className="shrink-0 rounded-full border border-border px-2.5 py-1 font-mono text-[11px] text-muted-foreground">
          {reviewed}/{segments.length} reviewed
        </span>
      </div>

      <ul className="mt-4 space-y-2">
        {segments.map((segment, i) => (
          <SegmentRow
            key={segment.id}
            segment={segment}
            index={i}
            expanded={segment.id === selectedId}
            judgement={judgements[segment.id]}
            onToggle={() => onSelect(segment.id === selectedId ? null : segment.id)}
            onJudge={(j) => onJudge(segment.id, j)}
          />
        ))}
      </ul>
    </div>
  );
}
