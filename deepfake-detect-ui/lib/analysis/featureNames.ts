const FRIENDLY_NAMES: Record<string, string> = {
  f0: "F0 (pitch)",
  voicing_confidence: "Voicing confidence",
  spectral_centroid: "Spectral centroid",
  spectral_bandwidth: "Spectral bandwidth",
  spectral_rolloff: "Spectral roll-off",
  spectral_flatness: "Spectral flatness",
  zero_crossing_rate: "Zero-crossing rate",
  short_time_energy: "Short-time energy",
};

/** Maps src/models/acoustic/features.py FEATURE_NAMES to readable labels. */
export function friendlyName(feature: string): string {
  if (FRIENDLY_NAMES[feature]) return FRIENDLY_NAMES[feature];
  const mfccMatch = feature.match(/^mfcc(_delta2?)?_(\d+)$/);
  if (mfccMatch) {
    const kind =
      mfccMatch[1] === "_delta2" ? "MFCC ΔΔ" : mfccMatch[1] === "_delta" ? "MFCC Δ" : "MFCC";
    return `${kind} ${mfccMatch[2]}`;
  }
  return feature;
}
