import { Sparkles } from "lucide-react";
import { SCENARIOS } from "@/lib/mock/scenarios";
import type { InferenceResult } from "@/lib/types";

const MODALITY_LABEL: Record<InferenceResult["manipulatedModalityGuess"], string> = {
  none: "No manipulated modality detected",
  video: "Visual manipulation implicated",
  audio: "Acoustic manipulation implicated",
  both: "Both modalities implicated",
};

export function NarrativeSummary({ result }: { result: InferenceResult }) {
  const def = SCENARIOS[result.scenario];

  return (
    <div className="glass-panel rounded-2xl p-5">
      <div className="flex items-center gap-2">
        <Sparkles className="h-4 w-4 text-cyan" />
        <h3 className="text-sm font-semibold text-foreground">Summary</h3>
      </div>
      <p className="mt-2 text-[13px] leading-relaxed text-muted-foreground">{def.narrative}</p>
      <p className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-white/10 px-2.5 py-1 text-[11px] text-muted-foreground">
        {MODALITY_LABEL[result.manipulatedModalityGuess]}
      </p>
    </div>
  );
}
