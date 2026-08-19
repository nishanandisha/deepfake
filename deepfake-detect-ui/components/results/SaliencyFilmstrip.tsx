"use client";

import { useState } from "react";
import { Layers } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import type { SaliencyFrame } from "@/lib/types";
import { cn } from "@/lib/utils";

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function SaliencyFilmstrip({ frames }: { frames: SaliencyFrame[] }) {
  const [selected, setSelected] = useState<SaliencyFrame | null>(null);
  const [overlayOn, setOverlayOn] = useState(true);

  const topFrame = [...frames].sort((a, b) => b.salienceScore - a.salienceScore)[0];

  return (
    <div className="glass-panel rounded-2xl p-5">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Visual saliency</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Grad-CAM overlay over extracted frames — brighter regions drove the visual score most.
          </p>
        </div>
        {topFrame && (
          <span className="hidden shrink-0 rounded-full border border-white/10 px-2.5 py-1 font-mono text-[11px] text-muted-foreground sm:block">
            peak @ {formatTime(topFrame.timestamp)}
          </span>
        )}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2.5 sm:grid-cols-6">
        {frames.map((frame) => (
          <button
            key={frame.frameIndex}
            type="button"
            onClick={() => setSelected(frame)}
            className="group relative aspect-video overflow-hidden rounded-lg border border-white/10 transition hover:border-cyan/50"
          >
            <img src={frame.thumbnailUrl} alt={`Frame at ${formatTime(frame.timestamp)}`} className="h-full w-full object-cover" />
            <img
              src={frame.heatmapUrl}
              alt=""
              className="absolute inset-0 h-full w-full object-cover"
            />
            <span className="absolute bottom-1 right-1 rounded bg-black/60 px-1 font-mono text-[9px] text-white/90">
              {formatTime(frame.timestamp)}
            </span>
            {frame.salienceScore > 0.55 && (
              <span className="absolute left-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-block/80">
                <Layers className="h-2.5 w-2.5 text-white" />
              </span>
            )}
          </button>
        ))}
      </div>

      <Dialog open={!!selected} onOpenChange={(open) => !open && setSelected(null)}>
        <DialogContent className="sm:max-w-lg">
          {selected && (
            <>
              <DialogTitle>Frame at {formatTime(selected.timestamp)}</DialogTitle>
              <DialogDescription>
                Salience score {Math.round(selected.salienceScore * 100)}% — how strongly this frame
                contributed to the visual branch&apos;s prediction.
              </DialogDescription>
              <div className="relative aspect-video overflow-hidden rounded-lg border border-white/10">
                <img src={selected.thumbnailUrl} alt="" className="h-full w-full object-cover" />
                {overlayOn && (
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
                    ? "border-cyan/40 bg-cyan/10 text-cyan"
                    : "border-white/10 text-muted-foreground hover:text-foreground"
                )}
              >
                {overlayOn ? "Hide saliency overlay" : "Show saliency overlay"}
              </button>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
