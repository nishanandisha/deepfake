"use client";

import { useEffect, useState } from "react";
import { Cpu, FlaskConical, Loader2 } from "lucide-react";
import { isBackendAvailable } from "@/lib/api/inference";

/**
 * States plainly whether results come from the trained model or the mock
 * engine.
 *
 * This matters: the mock produces confident, plausible-looking output, and
 * a demo that silently falls back would misrepresent a real detection as
 * having come from the model. The distinction is never left implicit.
 */
export function BackendBadge() {
  const [state, setState] = useState<"checking" | "model" | "mock">("checking");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const live = await isBackendAvailable();
      if (!cancelled) setState(live ? "model" : "mock");
    };
    check();
    const timer = setInterval(check, 15000); // pick up the server coming up
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  if (state === "checking") {
    return (
      <span className="flex items-center gap-1.5 rounded-full border border-white/10 px-2.5 py-1 font-mono text-[11px] text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        checking
      </span>
    );
  }

  const isModel = state === "model";
  return (
    <span
      title={
        isModel
          ? "Results come from the trained cross-attention fusion model."
          : "Backend offline - showing simulated results, not model output."
      }
      className={
        "flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[11px] " +
        (isModel
          ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
          : "border-amber-500/30 bg-amber-500/10 text-amber-300")
      }
    >
      {isModel ? <Cpu className="h-3 w-3" /> : <FlaskConical className="h-3 w-3" />}
      {isModel ? "live model" : "mock data"}
    </span>
  );
}
