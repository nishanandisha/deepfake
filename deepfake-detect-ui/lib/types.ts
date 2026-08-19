export type Decision = "approve" | "flag" | "block";

export type ManipulatedModality = "video" | "audio" | "both" | "none";

export type ScenarioId = "authentic" | "fake_video" | "fake_audio" | "fake_both";

export interface AcousticShapEntry {
  feature: string;
  value: number; // signed SHAP value; positive pushes toward "fake"
}

export interface SaliencyFrame {
  frameIndex: number;
  timestamp: number; // seconds into clip
  salienceScore: number; // 0..1, how much this frame drove the decision
  thumbnailUrl: string; // real, extracted from the uploaded file (object/data URL)
  heatmapUrl: string; // synthetic Grad-CAM-style overlay, canvas-generated
}

export interface WaveformData {
  peaks: number[]; // normalized -1..1 downsampled peaks
  durationSeconds: number;
  suspiciousRegions: { startSeconds: number; endSeconds: number; intensity: number }[];
}

export interface InferenceResult {
  sampleId: string;
  fileName: string;
  createdAt: string; // ISO timestamp

  // src/evaluation/policy.py: decide()
  decision: Decision;
  cScore: number; // calibrated authenticity, 1 - sigmoid(logit / T)
  tauLo: number;
  tauHi: number;

  // src/models/fusion/cross_attention.py forward() output
  gate: number; // visual weight in [0,1]; 1-gate = acoustic weight
  yHatVisual: number; // branch fake-probability, aux head
  yHatAcoustic: number;
  yHatFused: number;

  acousticShap: AcousticShapEntry[];
  visualSaliency: SaliencyFrame[];
  waveform: WaveformData | null;

  falseSuppressionRate: number;
  reviewQueueRate: number;
  manipulatedModalityGuess: ManipulatedModality;

  scenario: ScenarioId;
}

export interface QueueItem {
  sampleId: string;
  fileName: string;
  decision: Decision;
  cScore: number;
  createdAt: string;
}

export type PipelineStageId =
  | "preprocessing"
  | "visual_branch"
  | "acoustic_branch"
  | "fusion"
  | "calibration"
  | "explanation";

export interface PipelineStage {
  id: PipelineStageId;
  label: string;
  detail: string;
}
