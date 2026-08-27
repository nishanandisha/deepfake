"use client";

import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { FileVideo2, Upload, X } from "lucide-react";
import { usePipelineStore } from "@/store/pipelineStore";
import { cn } from "@/lib/utils";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

export function Dropzone() {
  const { file, previewUrl, setFile } = usePipelineStore();
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const acceptFile = useCallback(
    (candidate: File | undefined) => {
      if (!candidate) return;
      if (!candidate.type.startsWith("video/") && !candidate.type.startsWith("audio/")) return;
      setFile(candidate);
    },
    [setFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      acceptFile(e.dataTransfer.files?.[0]);
    },
    [acceptFile]
  );

  if (file && previewUrl) {
    return (
      <div className="glass-panel rounded-2xl p-4 sm:p-5">
        <div className="flex items-start gap-4">
          <div className="relative shrink-0 overflow-hidden rounded-xl border border-glass-border bg-background">
            {file.type.startsWith("video/") ? (
              <video
                src={previewUrl}
                className="h-24 w-40 object-cover"
                muted
                playsInline
                preload="metadata"
              />
            ) : (
              <div className="flex h-24 w-40 items-center justify-center">
                <FileVideo2 className="h-8 w-8 text-brand" />
              </div>
            )}
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate font-medium text-foreground">{file.name}</p>
            <p className="mt-1 font-mono text-xs text-muted-foreground">
              {formatBytes(file.size)} &middot; {file.type || "unknown type"}
            </p>
            <button
              type="button"
              onClick={() => setFile(null)}
              className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-glass-border px-2.5 py-1 text-xs text-muted-foreground transition hover:border-destructive/40 hover:text-destructive"
            >
              <X className="h-3.5 w-3.5" />
              Remove
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
      className={cn(
        "glass-panel group relative flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-14 text-center transition-colors",
        isDragging ? "border-brand/70 bg-brand/5" : "border-border hover:border-input"
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*,audio/*"
        className="hidden"
        onChange={(e) => acceptFile(e.target.files?.[0])}
      />
      <motion.div
        animate={{ y: isDragging ? -4 : 0 }}
        className="mb-4 flex h-14 w-14 items-center justify-center rounded-full accent-gradient-bg"
      >
        <Upload className="h-6 w-6 text-background" />
      </motion.div>
      <p className="text-sm font-medium text-foreground">
        Drop a video or audio clip, or click to browse
      </p>
      <p className="mt-1.5 text-xs text-muted-foreground">
        MP4, MOV, WAV or MP3 — the picture and the sound are checked separately.
      </p>
    </div>
  );
}
