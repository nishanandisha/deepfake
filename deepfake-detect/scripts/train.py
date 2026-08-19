"""Training entry point. Dispatches to the branch-specific trainer based on
cfg.model.name (set by `model=visual|acoustic|fusion` on the CLI).
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger
from src.utils.seed import set_seed

TRAINERS = {
    "visual_branch": "src.training.train_visual:train_visual",
    "acoustic_branch": "src.training.train_acoustic:train_acoustic",
    "fusion": "src.training.train_fusion:train_fusion",
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    logger = get_logger(cfg.experiment_name, log_dir=Path(cfg.output_dir))
    logger.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.model.name not in TRAINERS:
        logger.info(
            f"No trainer implemented yet for model={cfg.model.name!r} -- see the "
            "build plan for the relevant stage (train_acoustic.py / train_fusion.py)."
        )
        return

    module_path, func_name = TRAINERS[cfg.model.name].split(":")
    import importlib

    train_fn = getattr(importlib.import_module(module_path), func_name)
    train_fn(cfg, logger=logger)


if __name__ == "__main__":
    main()
