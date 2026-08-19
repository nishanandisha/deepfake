"use client";

import { Check } from "lucide-react";
import { SCENARIOS, SCENARIO_ORDER } from "@/lib/mock/scenarios";
import { usePipelineStore } from "@/store/pipelineStore";
import { cn } from "@/lib/utils";

const DOT_CLASS: Record<string, string> = {
  authentic: "bg-approve",
  fake_video: "bg-flag",
  fake_audio: "bg-flag",
  fake_both: "bg-block",
};

export function ScenarioPicker() {
  const { scenario, setScenario } = usePipelineStore();

  return (
    <div>
      <p className="mb-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        Simulated ground truth
      </p>
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
                  ? "border-cyan/50 bg-cyan/[0.06] shadow-[0_0_0_1px_rgba(34,211,238,0.15)]"
                  : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", DOT_CLASS[id])} />
                  <span className="text-sm font-medium text-foreground">{def.label}</span>
                </div>
                {active && <Check className="h-3.5 w-3.5 text-cyan" />}
              </div>
              <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{def.description}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
