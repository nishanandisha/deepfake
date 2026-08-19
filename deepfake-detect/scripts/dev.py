"""Cross-platform task runner (Windows has no `make` by default).

Mirrors the Makefile targets: setup, lint, test, train-visual, train-acoustic,
train-fusion, evaluate. Usage: python scripts/dev.py <task>
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "setup": [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    "lint": [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"],
    "test": [sys.executable, "-m", "pytest"],
    "download-data": [sys.executable, "scripts/download_data.py", "--output-dir", "data/raw/lavdf"],
    "inspect-data": [
        sys.executable, "scripts/inspect_dataset.py", "--root", "data/raw/lavdf", "--full-index",
    ],
    "build-splits": [sys.executable, "scripts/build_splits.py"],
    "warm-cache": [sys.executable, "scripts/warm_cache.py"],
    "train-visual": [sys.executable, "scripts/train.py", "model=visual"],
    "train-acoustic": [sys.executable, "scripts/train.py", "model=acoustic"],
    "train-fusion": [sys.executable, "scripts/train.py", "model=fusion"],
    "evaluate": [sys.executable, "scripts/eval.py"],
    "evaluate-late-fusion": [sys.executable, "scripts/evaluate_late_fusion.py"],
    "calibrate": [sys.executable, "scripts/calibrate.py"],
    "explain": [sys.executable, "scripts/explain.py"],
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in TASKS:
        print(f"Usage: python scripts/dev.py <{'|'.join(TASKS)}>")
        sys.exit(1)

    cmd = TASKS[sys.argv[1]]
    result = subprocess.run(cmd, cwd=ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
