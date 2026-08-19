"""Runs the whole trained pipeline end to end, in dependency order.

  visual branch -> acoustic branch -> late-fusion baseline
  -> cross-attention fusion -> calibration/policy -> explanations

Each stage is a separate subprocess so a crash in one is isolated and
reported rather than taking down the rest, and so every stage's own
resume-from-checkpoint logic still applies -- re-running this script after
an interruption picks up where it stopped instead of retraining.

Usage:
  python scripts/run_pipeline.py --data lean --training lean \\
      --output-root D:/deepfake-data/outputs --splits-dir D:/deepfake-data/splits_lean \\
      --cache-dir D:/deepfake-data/cache_lean
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(name: str, command: list, log_path: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===\n  {' '.join(command[1:])}", flush=True)

    # NOTE: expandable_segments is a no-op on Windows -- PyTorch warns
    # "not supported on this platform" and ignores it. Left set because it
    # does help on Linux, but on this machine VRAM headroom comes from
    # batch_size and num_workers instead: each DataLoader worker is a
    # spawned process, and raising workers 4 -> 8 was what pushed a
    # previously-working run into OOM on a 4GB card.
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env
        )
    elapsed = time.time() - started

    tail = ""
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-12:])

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  {status} in {elapsed/60:.1f} min")
    if result.returncode != 0:
        print(f"  --- last lines of {log_path} ---\n{tail}")

    return {"stage": name, "ok": result.returncode == 0, "minutes": elapsed / 60,
            "log": str(log_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline.")
    parser.add_argument("--data", default="lean")
    parser.add_argument("--training", default="lean")
    parser.add_argument("--output-root", default="D:/deepfake-data/outputs")
    parser.add_argument("--splits-dir", default="D:/deepfake-data/splits_lean")
    parser.add_argument("--cache-dir", default="D:/deepfake-data/cache_lean")
    parser.add_argument("--hydra-dir", default="D:/deepfake-data/hydra")
    parser.add_argument(
        "--skip", default="", help="comma-separated stage names to skip (e.g. explain)"
    )
    args = parser.parse_args()

    out = args.output_root.rstrip("/")
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    py = sys.executable

    common = [
        f"data={args.data}",
        f"training={args.training}",
        f"data.splits_dir={args.splits_dir}",
        f"data.cache_dir={args.cache_dir}",
    ]

    stages = [
        ("visual", [py, "-W", "ignore", "scripts/train.py", "model=visual",
                    f"output_dir={out}/visual", *common,
                    f"hydra.run.dir={args.hydra_dir}/visual"]),
        ("acoustic", [py, "-W", "ignore", "scripts/train.py", "model=acoustic",
                      f"output_dir={out}/acoustic", *common,
                      f"hydra.run.dir={args.hydra_dir}/acoustic"]),
        ("late_fusion", [py, "-W", "ignore", "scripts/evaluate_late_fusion.py",
                         f"output_dir={out}/late_fusion",
                         f"visual_checkpoint={out}/visual/checkpoints/best.pt",
                         f"acoustic_checkpoint={out}/acoustic/checkpoints/best.pt",
                         f"data={args.data}", f"data.splits_dir={args.splits_dir}",
                         f"data.cache_dir={args.cache_dir}",
                         f"hydra.run.dir={args.hydra_dir}/late_fusion"]),
        ("fusion", [py, "-W", "ignore", "scripts/train.py", "model=fusion",
                    f"output_dir={out}/fusion", *common,
                    f"visual_checkpoint={out}/visual/checkpoints/best.pt",
                    f"acoustic_checkpoint={out}/acoustic/checkpoints/best.pt",
                    f"late_fusion_results={out}/late_fusion/results.json",
                    f"hydra.run.dir={args.hydra_dir}/fusion"]),
        ("calibration", [py, "-W", "ignore", "scripts/calibrate.py",
                         f"output_dir={out}/calibration",
                         f"fusion_checkpoint={out}/fusion/checkpoints/best.pt",
                         f"data={args.data}", f"data.splits_dir={args.splits_dir}",
                         f"data.cache_dir={args.cache_dir}",
                         f"hydra.run.dir={args.hydra_dir}/calibration"]),
        ("explain", [py, "-W", "ignore", "scripts/explain.py",
                     f"output_dir={out}/explain",
                     f"fusion_checkpoint={out}/fusion/checkpoints/best.pt",
                     f"policy_json={out}/calibration/policy.json",
                     f"data={args.data}", f"data.splits_dir={args.splits_dir}",
                     f"data.cache_dir={args.cache_dir}",
                     f"hydra.run.dir={args.hydra_dir}/explain"]),
    ]

    results = []
    for name, command in stages:
        if name in skip:
            print(f"\n=== {name} === skipped")
            continue

        result = _run(name, command, Path(out) / "logs" / f"{name}.log")
        results.append(result)
        if not result["ok"]:
            # Later stages consume this one's checkpoint, so continuing
            # would only produce confusing downstream failures.
            print(f"\nStopping: {name} failed. Fix it and re-run "
                  "(completed stages resume from their checkpoints).")
            break

    summary_path = Path(out) / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['stage']:12s} {'OK' if r['ok'] else 'FAILED':7s} {r['minutes']:.1f} min")
    print(f"\nTotal: {sum(r['minutes'] for r in results):.1f} min")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
