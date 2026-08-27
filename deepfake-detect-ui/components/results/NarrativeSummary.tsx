import { Ear, Eye, ScanSearch } from "lucide-react";
import type { InferenceResult, TamperSegment } from "@/lib/types";
import { buildSummary, tamperedDuration, clipDuration, isAudioOnly } from "@/lib/analysis/localization";
import { cn } from "@/lib/utils";

function TrackChip({
  icon: Icon,
  label,
  implicated,
  seconds,
  duration,
  absent,
}: {
  icon: typeof Eye;
  label: string;
  implicated: boolean;
  seconds: number;
  duration: number;
  absent?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px]",
        absent
          ? "border-border bg-secondary/40 text-muted-foreground"
          : implicated
            ? "border-deepfake/40 bg-deepfake/10 text-deepfake"
            : "border-authentic/35 bg-authentic/[0.07] text-authentic"
      )}
    >
      <Icon className="h-3 w-3" />
      {label}:{" "}
      {absent
        ? "no video track in this file"
        : implicated
          ? `${seconds.toFixed(1)}s of ${duration.toFixed(1)}s altered`
          : "no manipulation found"}
    </span>
  );
}

export function NarrativeSummary({
  result,
  segments,
}: {
  result: InferenceResult;
  segments: TamperSegment[];
}) {
  const duration = clipDuration(result);
  const videoSeconds = tamperedDuration(segments, "video");
  const audioSeconds = tamperedDuration(segments, "audio");

  return (
    <div className="glass-panel rounded-2xl p-5">
      <div className="flex items-center gap-2">
        <ScanSearch className="h-4 w-4 text-brand" />
        <h3 className="text-sm font-semibold text-foreground">What the model found</h3>
      </div>

      <p className="mt-2 text-[13px] leading-relaxed text-muted-foreground">
        {buildSummary(result, segments)}
      </p>

      <div className="mt-3 flex flex-wrap gap-2">
        <TrackChip
          icon={Eye}
          label="Video"
          implicated={videoSeconds > 0}
          seconds={videoSeconds}
          duration={duration}
          absent={isAudioOnly(result)}
        />
        <TrackChip
          icon={Ear}
          label="Audio"
          implicated={audioSeconds > 0}
          seconds={audioSeconds}
          duration={duration}
        />
      </div>
    </div>
  );
}
