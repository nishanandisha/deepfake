"use client";

import { Check } from "lucide-react";
import { SCENARIOS, SCENARIO_ORDER } from "@/lib/mock/scenarios";
import { usePipelineStore } from "@/store/pipelineStore";
import { cn } from "@/lib/utils";

const DOT_CLASS: Record<string, string> = {
  authentic: "bg-authentic",
  fake_video: "bg-uncertain",
  fake_audio: "bg-uncertain",
  fake_both: "bg-deepfake",
};

export function ScenarioPicker() {
  const { scenario, setScenario } = usePipelineStore();

  return (
    <div>
      <div className="mb-3 flex items-baseline gap-2">
        <p className="text-xs font-medium uppercase tracking-wide text-uncertain">
          Mock mode — pick what to simulate
        </p>
        <p className="text-[11px] text-muted-foreground">
          the detection backend is offline
        </p>
      </div>
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        {SCENARIO_ORDER.map((id) => {
          const def = SCENARIOS[id];
          const active = scenario === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => setScenario(id)}
              className={cn(
                "relative rounded-xl border px-4 py-3 text-left transition-all",
                active
                  ? "border-brand/50 bg-brand/[0.06] shadow-[0_0_0_1px_rgba(50,130,184,0.22)]"
                  : "border-border bg-secondary/25 hover:border-input hover:bg-secondary/50"
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", DOT_CLASS[id])} />
                  <span className="text-sm font-medium text-foreground">{def.label}</span>
                </div>
                {active && <Check className="h-3.5 w-3.5 text-brand" />}
              </div>
              <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{def.description}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
