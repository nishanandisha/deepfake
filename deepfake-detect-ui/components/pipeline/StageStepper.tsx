"use client";

import { motion } from "framer-motion";
import { Check, Loader2 } from "lucide-react";
import { PIPELINE_STAGES } from "@/lib/mock/pipelineStages";
import { cn } from "@/lib/utils";

export function StageStepper({ stageIndex }: { stageIndex: number }) {
  return (
    <div className="glass-panel w-full max-w-md rounded-2xl p-6">
      <ol className="space-y-1">
        {PIPELINE_STAGES.map((stage, i) => {
          const state = i < stageIndex ? "done" : i === stageIndex ? "active" : "pending";
          return (
            <li key={stage.id} className="flex items-start gap-3 py-2.5">
              <span
                className={cn(
                  "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[10px] transition-colors",
                  state === "done" && "border-authentic/50 bg-authentic/15 text-authentic",
                  state === "active" && "border-brand/50 bg-brand/15 text-brand",
                  state === "pending" && "border-border text-muted-foreground"
                )}
              >
                {state === "done" && <Check className="h-3 w-3" />}
                {state === "active" && <Loader2 className="h-3 w-3 animate-spin" />}
                {state === "pending" && <span className="h-1 w-1 rounded-full bg-current" />}
              </span>
              <div className="min-w-0">
                <p
                  className={cn(
                    "text-sm font-medium transition-colors",
                    state === "pending" ? "text-muted-foreground" : "text-foreground"
                  )}
                >
                  {stage.label}
                </p>
                {state === "active" && (
                  <motion.p
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    className="mt-0.5 text-xs text-muted-foreground"
                  >
                    {stage.detail}
                  </motion.p>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
