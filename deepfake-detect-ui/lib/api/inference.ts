import type { InferenceResult, ScenarioId } from "@/lib/types";

/**
 * Client for the real Python backend (deepfake-detect/scripts/serve.py).
 *
 * Mirrors runMockInference's signature exactly so store/pipelineStore.ts can
 * swap between them without touching component code.
 */

const BASE_URL = process.env.NEXT_PUBLIC_INFERENCE_API ?? "http://localhost:8000";

export interface RealRunOptions {
  file: File;
  scenario: ScenarioId; // ignored by the real model; kept for signature parity
  onStage?: (stageIndex: number) => void;
}

export async function isBackendAvailable(timeoutMs = 1500): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch(`${BASE_URL}/api/health`, { signal: controller.signal });
    clearTimeout(timer);
    return response.ok;
  } catch {
    return false;
  }
}

export async function runRealInference({
  file,
  onStage,
}: RealRunOptions): Promise<InferenceResult> {
  const body = new FormData();
  body.append("file", file);

  // The backend runs all six stages in a single request, so there are no
  // intermediate events to subscribe to. Advance the stepper on a timer
  // that reflects the real relative cost of each stage (SHAP dominates),
  // and stop as soon as the response lands.
  const STAGE_DELAYS_MS = [200, 900, 700, 500, 200, 1500];
  let stage = 0;
  onStage?.(0);
  let timer: ReturnType<typeof setTimeout> | undefined;
  const advance = () => {
    if (stage < STAGE_DELAYS_MS.length - 1) {
      stage += 1;
      onStage?.(stage);
      timer = setTimeout(advance, STAGE_DELAYS_MS[stage]);
    }
  };
  timer = setTimeout(advance, STAGE_DELAYS_MS[0]);

  try {
    const response = await fetch(`${BASE_URL}/api/infer`, { method: "POST", body });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(
        `Inference failed (${response.status}). ${detail.slice(0, 200)}`.trim()
      );
    }
    const result = (await response.json()) as InferenceResult;

    onStage?.(STAGE_DELAYS_MS.length - 1);
    return result;
  } finally {
    if (timer) clearTimeout(timer);
  }
}
