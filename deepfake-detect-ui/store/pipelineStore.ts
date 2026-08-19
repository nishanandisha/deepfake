"use client";

import { create } from "zustand";
import { isBackendAvailable, runRealInference } from "@/lib/api/inference";
import { runMockInference } from "@/lib/mock/engine";
import { QUEUE_SEED } from "@/lib/mock/queueSeed";
import { PIPELINE_STAGES } from "@/lib/mock/pipelineStages";
import type { InferenceResult, QueueItem, ScenarioId } from "@/lib/types";

export type RunStatus = "idle" | "running" | "done" | "error";
export type BackendMode = "model" | "mock" | "unknown";

interface PipelineState {
  backendMode: BackendMode;
  file: File | null;
  previewUrl: string | null;
  scenario: ScenarioId;
  status: RunStatus;
  stageIndex: number;
  result: InferenceResult | null;
  error: string | null;
  queue: QueueItem[];

  setFile: (file: File | null) => void;
  setScenario: (scenario: ScenarioId) => void;
  runInference: () => Promise<void>;
  reset: () => void;
}

export const usePipelineStore = create<PipelineState>((set, get) => ({
  backendMode: "unknown",
  file: null,
  previewUrl: null,
  scenario: "authentic",
  status: "idle",
  stageIndex: -1,
  result: null,
  error: null,
  queue: QUEUE_SEED,

  setFile: (file) => {
    const prev = get().previewUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      file,
      previewUrl: file ? URL.createObjectURL(file) : null,
      status: "idle",
      result: null,
      error: null,
      stageIndex: -1,
    });
  },

  setScenario: (scenario) => set({ scenario }),

  runInference: async () => {
    const { file, scenario } = get();
    if (!file) return;
    set({ status: "running", stageIndex: 0, error: null, result: null });
    try {
      // Prefer the real model when scripts/serve.py is up; fall back to the
      // mock engine so the UI still demos without the Python backend running.
      const useRealBackend = await isBackendAvailable();
      const run = useRealBackend ? runRealInference : runMockInference;
      const result = await run({
        file,
        scenario,
        onStage: (stageIndex) => set({ stageIndex }),
      });
      set({ backendMode: useRealBackend ? "model" : "mock" });
      set((state) => ({
        status: "done",
        stageIndex: PIPELINE_STAGES.length - 1,
        result,
        queue: [
          {
            sampleId: result.sampleId,
            fileName: result.fileName,
            decision: result.decision,
            cScore: result.cScore,
            createdAt: result.createdAt,
          },
          ...state.queue,
        ].slice(0, 12),
      }));
    } catch (err) {
      set({ status: "error", error: err instanceof Error ? err.message : "Analysis failed" });
    }
  },

  reset: () => {
    const prev = get().previewUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({ file: null, previewUrl: null, status: "idle", stageIndex: -1, result: null, error: null });
  },
}));
