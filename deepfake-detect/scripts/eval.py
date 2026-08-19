"""Evaluation entry point (placeholder). See Stage 8 for the full eval suite."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    logger = get_logger(f"{cfg.experiment_name}_eval", log_dir=Path(cfg.output_dir))
    logger.info("Evaluation pipeline not implemented yet -- see Stage 8.")


if __name__ == "__main__":
    main()
