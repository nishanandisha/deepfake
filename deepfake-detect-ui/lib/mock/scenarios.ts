import type { AcousticShapEntry, Decision, ManipulatedModality, ScenarioId } from "@/lib/types";

export interface ScenarioDefinition {
  id: ScenarioId;
  label: string;
  description: string;
  manipulatedModalityGuess: ManipulatedModality;
  /** Base authenticity score before small run-to-run jitter (0..1, 1 = authentic). */
  cScoreBase: number;
  cScoreJitter: number;
  gateBase: number; // visual weight base
  gateJitter: number;
  yHatVisualBase: number;
  yHatAcousticBase: number;
  acousticShap: AcousticShapEntry[];
  narrative: string;
}

// tau thresholds from a representative Stage 6 calibration run
// (src/evaluation/policy.py: approve if c >= tauHi, block if c < tauLo).
export const TAU_LO = 0.35;
export const TAU_HI = 0.72;
export const FALSE_SUPPRESSION_RATE = 0.018;
export const REVIEW_QUEUE_RATE = 0.14;

export function decisionForScore(cScore: number): Decision {
  if (cScore >= TAU_HI) return "approve";
  if (cScore < TAU_LO) return "block";
  return "flag";
}

// Feature names mirror src/models/acoustic/features.py FEATURE_NAMES exactly,
// so the UI speaks the same vocabulary the real SHAP explanations will use.
export const SCENARIOS: Record<ScenarioId, ScenarioDefinition> = {
  authentic: {
    id: "authentic",
    label: "Authentic clip",
    description: "Unmanipulated audio and video — should clear review automatically.",
    manipulatedModalityGuess: "none",
    cScoreBase: 0.91,
    cScoreJitter: 0.04,
    gateBase: 0.52,
    gateJitter: 0.06,
    yHatVisualBase: 0.08,
    yHatAcousticBase: 0.1,
    acousticShap: [
      { feature: "f0", value: 0.02 },
      { feature: "spectral_flatness", value: -0.03 },
      { feature: "voicing_confidence", value: 0.01 },
      { feature: "zero_crossing_rate", value: -0.02 },
      { feature: "mfcc_3", value: 0.015 },
      { feature: "spectral_centroid", value: -0.01 },
      { feature: "short_time_energy", value: 0.01 },
      { feature: "mfcc_delta_1", value: -0.008 },
    ],
    narrative:
      "Both branches agree the content is consistent with authentic capture. No acoustic descriptor or visual region drove the score in a fake-leaning direction.",
  },
  fake_video: {
    id: "fake_video",
    label: "Fake video only",
    description: "Face-swapped/synthesized visuals over an untouched audio track.",
    manipulatedModalityGuess: "video",
    cScoreBase: 0.22,
    cScoreJitter: 0.05,
    gateBase: 0.78,
    gateJitter: 0.05,
    yHatVisualBase: 0.91,
    yHatAcousticBase: 0.14,
    acousticShap: [
      { feature: "f0", value: 0.04 },
      { feature: "voicing_confidence", value: 0.03 },
      { feature: "spectral_flatness", value: -0.02 },
      { feature: "zero_crossing_rate", value: 0.02 },
      { feature: "mfcc_1", value: -0.015 },
      { feature: "spectral_rolloff", value: 0.012 },
      { feature: "short_time_energy", value: -0.01 },
      { feature: "mfcc_delta2_4", value: 0.008 },
    ],
    narrative:
      "The visual branch drove this decision — the gate weighted spatial-temporal evidence heavily, consistent with a face-region manipulation. Acoustic descriptors stayed near baseline, matching an untouched audio track.",
  },
  fake_audio: {
    id: "fake_audio",
    label: "Fake audio only",
    description: "Synthesized/cloned voice over genuine, unmanipulated video.",
    manipulatedModalityGuess: "audio",
    cScoreBase: 0.52,
    cScoreJitter: 0.06,
    gateBase: 0.24,
    gateJitter: 0.05,
    yHatVisualBase: 0.13,
    yHatAcousticBase: 0.83,
    acousticShap: [
      { feature: "f0", value: 0.21 },
      { feature: "voicing_confidence", value: 0.18 },
      { feature: "spectral_flatness", value: 0.15 },
      { feature: "spectral_centroid", value: 0.11 },
      { feature: "mfcc_2", value: 0.09 },
      { feature: "zero_crossing_rate", value: 0.07 },
      { feature: "mfcc_delta_5", value: -0.05 },
      { feature: "short_time_energy", value: 0.04 },
    ],
    narrative:
      "This one lands in the review band on purpose: acoustic descriptors (pitch stability, voicing confidence, spectral flatness) show the synthetic-speech signature, but the untouched visual track keeps the fused score from dropping further. Routed to a moderator rather than auto-blocked.",
  },
  fake_both: {
    id: "fake_both",
    label: "Fake video + audio",
    description: "Fully synthesized clip — both modalities manipulated.",
    manipulatedModalityGuess: "both",
    cScoreBase: 0.08,
    cScoreJitter: 0.03,
    gateBase: 0.5,
    gateJitter: 0.08,
    yHatVisualBase: 0.94,
    yHatAcousticBase: 0.89,
    acousticShap: [
      { feature: "f0", value: 0.24 },
      { feature: "voicing_confidence", value: 0.2 },
      { feature: "spectral_flatness", value: 0.17 },
      { feature: "spectral_centroid", value: 0.13 },
      { feature: "mfcc_2", value: 0.1 },
      { feature: "zero_crossing_rate", value: 0.08 },
      { feature: "mfcc_1", value: 0.06 },
      { feature: "short_time_energy", value: 0.05 },
    ],
    narrative:
      "Both branches independently flag this clip and the gate stays balanced rather than collapsing onto one modality — strong, corroborating evidence of manipulation in both the visual and acoustic streams.",
  },
};

export const SCENARIO_ORDER: ScenarioId[] = ["authentic", "fake_video", "fake_audio", "fake_both"];
