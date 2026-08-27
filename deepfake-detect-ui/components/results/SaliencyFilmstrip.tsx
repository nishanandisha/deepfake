"use client";

import { useState } from "react";
import { ShieldAlert } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import type { SaliencyFrame, TamperSegment } from "@/lib/types";
import { formatTimecode } from "@/lib/analysis/localization";
import { cn } from "@/lib/utils";

export function SaliencyFilmstrip({
  frames,
  segments,
}: {
  frames: SaliencyFrame[];
  segments: TamperSegment[];
}) {
  const [selected, setSelected] = useState<SaliencyFrame | null>(null);
  const [overlayOn, setOverlayOn] = useState(true);

  const videoSegments = segments.filter((s) => s.modality === "video");
  const isImplicated = (frame: SaliencyFrame) =>
    videoSegments.some(
      (s) => frame.timestamp >= s.startSeconds && frame.timestamp <= s.endSeconds
    );

  const implicatedCount = frames.filter(isImplicated).length;

  return (
    <div className="glass-panel rounded-2xl p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Every sampled frame</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            The full set the visual branch looked at, in order. Flagged frames are the ones
            inside an implicated span — the rest are shown for comparison.
          </p>
        </div>
        <span className="hidden shrink-0 rounded-full border border-border px-2.5 py-1 font-mono text-[11px] text-muted-foreground sm:block">
          {implicatedCount}/{frames.length} flagged
        </span>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2.5 sm:grid-cols-6">
        {frames.map((frame) => {
          const flagged = isImplicated(frame);
          return (
            <button
              key={frame.frameIndex}
              type="button"
              onClick={() => setSelected(frame)}
              className={cn(
                "group relative aspect-video overflow-hidden rounded-lg border transition",
                flagged
                  ? "border-deepfake/60 hover:border-deepfake"
                  : "border-border opacity-60 hover:opacity-100"
              )}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={frame.thumbnailUrl} alt="" className="h-full w-full object-cover" />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={frame.heatmapUrl}
                alt=""
                className="absolute inset-0 h-full w-full object-cover"
              />
              <span className="absolute bottom-1 right-1 rounded bg-background/80 px-1 font-mono text-[9px] text-foreground">
                {formatTimecode(frame.timestamp)}
              </span>
              {flagged && (
                <span className="absolute left-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-deepfake">
                  <ShieldAlert className="h-2.5 w-2.5 text-deepfake-foreground" />
                </span>
              )}
            </button>
          );
        })}
      </div>

      <Dialog open={!!selected} onOpenChange={(open) => !open && setSelected(null)}>
        <DialogContent className="sm:max-w-lg">
          {selected && (
            <>
              <DialogTitle>
                Frame at {formatTimecode(selected.timestamp)}
                {isImplicated(selected) ? " — implicated" : ""}
              </DialogTitle>
              <DialogDescription>
                Salience {Math.round(selected.salienceScore * 100)}% — how strongly this frame
                drove the visual branch&apos;s prediction.
              </DialogDescription>
              <div className="relative aspect-video overflow-hidden rounded-lg border border-border">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={selected.thumbnailUrl} alt="" className="h-full w-full object-cover" />
                {overlayOn && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={selected.heatmapUrl}
                    alt=""
                    className="absolute inset-0 h-full w-full object-cover"
                  />
                )}
              </div>
              <button
                type="button"
                onClick={() => setOverlayOn((v) => !v)}
                className={cn(
                  "self-start rounded-md border px-3 py-1.5 text-xs font-medium transition",
                  overlayOn
                    ? "border-brand/40 bg-brand/10 text-brand"
                    : "border-border text-muted-foreground hover:text-foreground"
                )}
              >
                {overlayOn ? "Hide Grad-CAM overlay" : "Show Grad-CAM overlay"}
              </button>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
