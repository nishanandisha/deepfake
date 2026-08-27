import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import type { InferenceResult } from "@/lib/types";
import { deepfakeScore, isAudioOnly } from "@/lib/analysis/localization";

function Metric({
  label,
  value,
  hint,
  highlight,
}: {
  label: string;
  value: string;
  hint: string;
  highlight?: boolean;
}) {
  return (
    <div className="flex items-center justify-between border-b border-border py-2.5 last:border-0">
      <Tooltip>
        <TooltipTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground">
          {label}
          <Info className="h-3 w-3 opacity-60" />
        </TooltipTrigger>
        <TooltipContent side="left" className="max-w-60">
          {hint}
        </TooltipContent>
      </Tooltip>
      <span
        className={
          "font-mono text-xs font-medium " + (highlight ? "text-brand-soft" : "text-foreground")
        }
      >
        {value}
      </span>
    </div>
  );
}

export function PolicyPanel({ result }: { result: InferenceResult }) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  // The backend's thresholds are stated on the authenticity scale; the UI
  // reads in manipulation likelihood, so both flip.
  const deepfakeAbove = 1 - result.tauLo;
  const genuineBelow = 1 - result.tauHi;

  return (
    <div className="glass-panel rounded-2xl p-5">
      <h3 className="text-sm font-semibold text-foreground">How the call was made</h3>
      <p className="mt-1 text-xs text-muted-foreground">
        Post-hoc calibrated thresholds, tuned on the validation split.
      </p>

      {isAudioOnly(result) && (
        <p className="mt-3 rounded-lg border border-uncertain/30 bg-uncertain/[0.07] px-3 py-2 text-[11px] leading-relaxed text-uncertain">
          Audio-only file. These thresholds were fitted on the combined
          audio-visual score, so treat an audio-only verdict as less precisely
          calibrated than one backed by both tracks.
        </p>
      )}

      <div className="mt-3">
        <Metric
          label="This clip"
          value={pct(deepfakeScore(result))}
          hint="Manipulation likelihood for this clip, after temperature calibration."
          highlight
        />
        <Metric
          label="Called deepfake above"
          value={pct(deepfakeAbove)}
          hint="Clips scoring above this are reported as deepfakes without needing human review."
        />
        <Metric
          label="Called genuine below"
          value={pct(genuineBelow)}
          hint="Clips scoring below this are reported as genuine. Anything between the two thresholds is returned as uncertain for you to decide."
        />
        <Metric
          label="False-alarm rate"
          value={pct(result.falseSuppressionRate)}
          hint="How often genuine clips are wrongly reported as deepfakes, held under a 2% ceiling — the costlier error here."
        />
        <Metric
          label="Sent for human review"
          value={pct(result.reviewQueueRate)}
          hint="Share of clips landing between the thresholds, where the model defers to you."
        />
      </div>
    </div>
  );
}
