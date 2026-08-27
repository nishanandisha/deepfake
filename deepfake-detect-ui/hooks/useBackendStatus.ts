"use client";

import { useEffect, useState } from "react";
import { isBackendAvailable } from "@/lib/api/inference";

export type BackendStatus = "checking" | "model" | "mock";

/**
 * Polls scripts/serve.py so the UI can say plainly whether a result came from
 * the trained model or the mock engine — and hide the scenario picker, which
 * only means anything in mock mode.
 */
export function useBackendStatus(): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>("checking");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const live = await isBackendAvailable();
      if (!cancelled) setStatus(live ? "model" : "mock");
    };
    check();
    const timer = setInterval(check, 15000); // pick up the server coming up
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return status;
}
