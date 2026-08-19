"""Stage 6 entry point: temperature calibration + three-way decision
policy selection, using the frozen Stage 5 fusion checkpoint.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibrate_and_select_policy import run_calibration_and_policy
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="calibration")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    run_calibration_and_policy(cfg)


if __name__ == "__main__":
    main()
