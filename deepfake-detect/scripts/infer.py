"""Single-sample inference entry point (placeholder). See Stage 9."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    raise NotImplementedError(
        "Inference pipeline (src/inference/pipeline.py) will be implemented "
        "in Stage 9, wrapping preprocessing -> branches -> fusion -> "
        "calibration -> policy -> explanation."
    )


if __name__ == "__main__":
    main()
