"use client";

import { FileVideo2 } from "lucide-react";
import { usePipelineStore } from "@/store/pipelineStore";
import type { QueueItem } from "@/lib/types";
import { cn } from "@/lib/utils";

const DOT: Record<QueueItem["verdict"], string> = {
  authentic: "bg-authentic",
  uncertain: "bg-uncertain",
  deepfake: "bg-deepfake",
};

function timeAgo(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const hours = diffMs / 3_600_000;
  if (hours < 1) return `${Math.max(1, Math.round(hours * 60))}m ago`;
  if (hours < 24) return `${Math.round(hours)}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function Sidebar({ activeSampleId }: { activeSampleId?: string }) {
  const queue = usePipelineStore((s) => s.queue);

  return (
    <aside className="hidden w-72 shrink-0 border-r border-border bg-sidebar lg:block">
      <div className="sticky top-16 max-h-[calc(100vh-4rem)] overflow-y-auto scrollbar-thin p-4">
        <p className="mb-3 px-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          Recent scans
        </p>
        <ul className="space-y-1">
          {queue.map((item) => (
            <li
              key={item.sampleId}
              className={cn(
                "flex items-center gap-2.5 rounded-lg px-2.5 py-2 transition-colors",
                item.sampleId === activeSampleId ? "bg-brand/[0.10]" : "hover:bg-secondary/40"
              )}
            >
              <FileVideo2 className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-xs text-foreground">{item.fileName}</p>
                <p className="font-mono text-[10px] text-muted-foreground">{timeAgo(item.createdAt)}</p>
              </div>
              <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", DOT[item.verdict])} />
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
