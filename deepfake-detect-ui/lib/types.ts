/**
 * `Decision` is the raw policy output of the Python backend
 * (src/evaluation/policy.py). It is a moderation-queue routing label and is
 * kept as-is on the wire so the UI stays a faithful view of the model.
 *
 * `Verdict` is what the product actually answers: is this a deepfake or not.
 * Nothing user-facing renders a `Decision` — see lib/analysis/localization.ts
 * for the mapping.
 */
export type Decision = "approve" | "flag" | "block";

export type Verdict = "deepfake" | "uncertain" | "authentic";

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
  heatmapUrl: string; // Grad-CAM overlay
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
  /** Branch fake-probability from the aux head. null for audio-only uploads. */
  yHatVisual: number | null;
  yHatAcoustic: number;
  yHatFused: number;

  /**
   * False for audio-only uploads (voice notes). The backend then runs the
   * acoustic branch standalone rather than feeding fusion a blank video
   * track, so there is no visual score, no saliency and no video timeline.
   */
  hasVideo: boolean;

  acousticShap: AcousticShapEntry[];
  visualSaliency: SaliencyFrame[];
  waveform: WaveformData | null;

  falseSuppressionRate: number;
  reviewQueueRate: number;
  manipulatedModalityGuess: ManipulatedModality;

  scenario: ScenarioId;
}

/* -------------------------------------------------------------------------
   Localization — "where exactly was it faked"
   ------------------------------------------------------------------------- */

export type SegmentModality = "video" | "audio";

/**
 * A contiguous stretch of the clip the model implicates, in one modality.
 *
 * Built from evidence the backend already returns — per-frame Grad-CAM
 * salience for video, acoustic suspicious regions for audio — rather than
 * from any new model output. See lib/analysis/localization.ts.
 */
export interface TamperSegment {
  id: string;
  modality: SegmentModality;
  startSeconds: number;
  endSeconds: number;
  peakSeconds: number;
  /** 0..1. Frame/region evidence scaled by that branch's fake probability. */
  confidence: number;
  /** Video segments only: the frames that fall inside this span. */
  frames: SaliencyFrame[];
  /** Audio segments only: acoustic descriptors that pushed hardest toward fake. */
  topFeatures: string[];
}

/** A reviewer's per-segment judgement. */
export type SegmentJudgement = "agree" | "disagree";

export interface QueueItem {
  sampleId: string;
  fileName: string;
  verdict: Verdict;
  /** 0..1 likelihood the clip is manipulated (1 - authenticity). */
  deepfakeScore: number;
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
