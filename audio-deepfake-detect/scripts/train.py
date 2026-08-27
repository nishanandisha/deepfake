"""Training entry point for the acoustic branch.

  python scripts/train.py                 # default data/training presets
  python scripts/train.py data=lean       # laptop-scale preset
  python scripts/train.py training.max_epochs=5 data.batch_size=4
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_acoustic import train_acoustic
from src.utils.logging import get_logger
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    logger = get_logger(cfg.experiment_name, log_dir=Path(cfg.output_dir))
    logger.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")

    train_acoustic(cfg, logger=logger)


if __name__ == "__main__":
    main()
