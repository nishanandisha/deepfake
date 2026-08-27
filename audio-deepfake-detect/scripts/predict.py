"""Score one or more audio clips with the trained acoustic model.

Accepts .wav/.flac/.mp3 and video containers with muxed audio (.mp4/.mov/
.avi) -- the video track is never decoded, only the audio stream.

Usage:
  python scripts/predict.py samples/fake/000919.mp4
  python scripts/predict.py samples/real/*.mp4 --json out.json
  python scripts/predict.py clip.wav --policy outputs/calibration/policy.json
  python scripts/predict.py clip.wav --no-explain      # skip SHAP (much faster)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.loader import DEFAULT_MODEL_PATH, describe, load_acoustic_model
from src.inference.pipeline import AudioDeepfakePipeline, InferenceConfig


def _print_result(result: dict) -> None:
    verdict = "FAKE" if result["pFake"] >= 0.5 else "REAL"
    print(f"\n{result['fileName']}")
    print(f"  P(fake)        {result['pFake']:.4f}   -> {verdict}")
    print(f"  authenticity c {result['cScore']:.4f}   decision: {result['decision']}"
          f"{'' if result['calibrated'] else '  (UNCALIBRATED -- plain 0.5 cut)'}")
    print(f"  audio analysed {result['framesAnalysed']} frames "
          f"({result['waveform']['durationSeconds']:.2f}s of signal)")

    if result["acousticShap"]:
        print("  top acoustic features (SHAP, + pushes toward fake):")
        # Bars are scaled to the largest attribution in this list, not to a
        # fixed constant -- logit-space SHAP values have no natural upper
        # bound, so a fixed scale saturates and shows nothing.
        largest = max(abs(item["value"]) for item in result["acousticShap"]) or 1.0
        for item in result["acousticShap"]:
            bar = "#" * max(int(abs(item["value"]) / largest * 30), 1)
            print(f"    {item['feature']:>22s}  {item['value']:+.5f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio deepfake detection on single clips.")
    parser.add_argument("media", nargs="+", help="audio or video files to score")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="path to the joblib artefact")
    parser.add_argument("--policy", default=None,
                        help="policy.json from scripts/calibrate.py; without it the score "
                             "is uncalibrated and the decision is a plain 0.5 cut")
    parser.add_argument("--no-explain", action="store_true", help="skip the SHAP attribution")
    parser.add_argument("--shap-samples", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", default=None, help="also write full results to this path")
    args = parser.parse_args()

    print("Loading model...")
    for line in describe(load_acoustic_model(args.model)):
        print(f"  {line}")

    pipeline = AudioDeepfakePipeline(InferenceConfig(
        model_path=args.model,
        policy_json=args.policy,
        explain=not args.no_explain,
        shap_samples=args.shap_samples,
        top_k_features=args.top_k,
        device=args.device,
    ))

    results = []
    for path in args.media:
        if not Path(path).exists():
            print(f"\n{path}: not found, skipping")
            continue
        result = pipeline.infer(path)
        results.append(result)
        _print_result(result)

    if args.json and results:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {len(results)} result(s) to {args.json}")


if __name__ == "__main__":
    main()
