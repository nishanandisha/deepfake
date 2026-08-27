"use client";

import { Cpu, FlaskConical, Loader2 } from "lucide-react";
import { useBackendStatus } from "@/hooks/useBackendStatus";

/**
 * States plainly whether results come from the trained model or the mock
 * engine.
 *
 * This matters: the mock produces confident, plausible-looking output, and
 * a demo that silently falls back would misrepresent a simulated result as
 * having come from the model. The distinction is never left implicit.
 */
export function BackendBadge() {
  const status = useBackendStatus();

  if (status === "checking") {
    return (
      <span className="flex items-center gap-1.5 rounded-full border border-border px-2.5 py-1 font-mono text-[11px] text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        checking
      </span>
    );
  }

  const isModel = status === "model";
  return (
    <span
      title={
        isModel
          ? "Results come from the trained cross-attention fusion model."
          : "Backend offline — showing simulated results, not model output."
      }
      className={
        "flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[11px] " +
        (isModel
          ? "border-authentic/35 bg-authentic/10 text-authentic"
          : "border-uncertain/35 bg-uncertain/10 text-uncertain")
      }
    >
      {isModel ? <Cpu className="h-3 w-3" /> : <FlaskConical className="h-3 w-3" />}
      {isModel ? "live model" : "mock data"}
    </span>
  );
}
