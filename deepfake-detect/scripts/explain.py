"""Stage 7 entry point: SHAP + Grad-CAM explanations and the attribution
agreement rate, using the frozen Stage 5 fusion checkpoint and the Stage 6
policy.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.run_explanations import run_explanations
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="explain")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    run_explanations(cfg)


if __name__ == "__main__":
    main()
