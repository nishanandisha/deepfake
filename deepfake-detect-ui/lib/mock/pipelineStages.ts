import type { PipelineStage } from "@/lib/types";

// Mirrors the six pipeline stages from deepfake-detection-build-plan.md.
export const PIPELINE_STAGES: PipelineStage[] = [
  {
    id: "preprocessing",
    label: "Preprocessing",
    detail: "Face detection/alignment + frame sampling; audio resampled to 16kHz mono",
  },
  {
    id: "visual_branch",
    label: "Visual branch",
    detail: "CNN spatial encoder + Transformer temporal encoder over aligned frames",
  },
  {
    id: "acoustic_branch",
    label: "Acoustic branch",
    detail: "MFCC/F0/spectral descriptors + Transformer encoder over audio frames",
  },
  {
    id: "fusion",
    label: "Cross-modal fusion",
    detail: "Bidirectional cross-attention + learned gate mixing both streams",
  },
  {
    id: "calibration",
    label: "Calibration & policy",
    detail: "Temperature scaling + approve/flag/block threshold policy",
  },
  {
    id: "explanation",
    label: "Explanation",
    detail: "SHAP modality attribution + acoustic descriptors + Grad-CAM saliency",
  },
];
