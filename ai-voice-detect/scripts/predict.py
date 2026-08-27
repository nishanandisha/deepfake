"""Classify an audio file as human or AI-generated.

  python scripts/predict.py clip.wav --layer 6
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration import load_policy
from src.inference.predict import VoicePredictor

BAR = 34


def render(prediction) -> str:
    d = prediction.to_dict()
    lines = [f"\n{d['fileName']}"]

    if d["verdict"] == "abstain":
        lines.append(f"  ABSTAIN  {d['reason']}")
        lines.append(f"  duration {d['durationSeconds']}s  voiced {d['speechSeconds']}s")
        return "\n".join(lines)

    p = d["pAiGenerated"]
    filled = int(round(p * BAR))
    label = "AI-GENERATED" if d["verdict"] == "ai" else "HUMAN"
    lines.append(f"  P(AI)   {p:.4f}  [{'#'*filled}{'.'*(BAR-filled)}]")
    lines.append(f"  verdict {label}")
    lines.append(f"  audio   {d['durationSeconds']}s ({d['speechSeconds']}s voiced), {d['reason']}")

    if len(d["windows"]) > 1:
        lines.append("  windows:")
        for w in d["windows"]:
            mark = "#" * int(round(w["pAiGenerated"] * 20))
            lines.append(f"    {w['startSeconds']:6.1f}-{w['endSeconds']:6.1f}s  "
                         f"{w['pAiGenerated']:.4f} {mark}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--checkpoint", default="outputs/run/best.pt")
    parser.add_argument("--policy", default="outputs/run/policy.json")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    policy = load_policy(args.policy) if Path(args.policy).exists() else {}
    predictor = VoicePredictor(args.checkpoint, policy, layer=args.layer)

    results = [predictor.predict(f) for f in args.files]
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for r in results:
            print(render(r))
