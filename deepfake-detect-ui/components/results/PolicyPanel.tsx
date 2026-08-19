import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import type { InferenceResult } from "@/lib/types";

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="flex items-center justify-between border-b border-white/[0.05] py-2.5 last:border-0">
      <Tooltip>
        <TooltipTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground">
          {label}
          <Info className="h-3 w-3 opacity-60" />
        </TooltipTrigger>
        <TooltipContent side="left" className="max-w-56">
          {hint}
        </TooltipContent>
      </Tooltip>
      <span className="font-mono text-xs font-medium text-foreground">{value}</span>
    </div>
  );
}

export function PolicyPanel({ result }: { result: InferenceResult }) {
  return (
    <div className="glass-panel rounded-2xl p-5">
      <h3 className="text-sm font-semibold text-foreground">Decision policy</h3>
      <p className="mt-1 text-xs text-muted-foreground">
        Post-hoc calibrated thresholds, tuned on the validation split.
      </p>

      <div className="mt-3">
        <Metric
          label="Block threshold (τ_lo)"
          value={result.tauLo.toFixed(2)}
          hint="Authenticity scores below this are blocked automatically."
        />
        <Metric
          label="Approve threshold (τ_hi)"
          value={result.tauHi.toFixed(2)}
          hint="Authenticity scores at or above this are approved automatically."
        />
        <Metric
          label="False-suppression rate"
          value={`${(result.falseSuppressionRate * 100).toFixed(1)}%`}
          hint="Rate of authentic content wrongly blocked, held under a 2% ceiling — the costlier error for this platform."
        />
        <Metric
          label="Review queue rate"
          value={`${(result.reviewQueueRate * 100).toFixed(1)}%`}
          hint="Share of submissions landing in the flag band, awaiting moderator review."
        />
      </div>
    </div>
  );
}
