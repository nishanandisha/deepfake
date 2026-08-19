"""Single logging helper shared by every training/eval/infer script.

Wraps Python's stdlib logging plus a TensorBoard SummaryWriter. Both are
returned from get_logger() so callers have one call site to set up
console + file + scalar logging consistently across scripts.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, name: str, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(fmt)
            self.logger.addHandler(console)

            file_handler = logging.FileHandler(self.log_dir / "run.log")
            file_handler.setFormatter(fmt)
            self.logger.addHandler(file_handler)

        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def scalars(self, tags_to_values: dict, step: int) -> None:
        for tag, value in tags_to_values.items():
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.close()


def get_logger(name: str, log_dir: Optional[Path] = None) -> ExperimentLogger:
    log_dir = log_dir or Path("outputs") / name
    return ExperimentLogger(name=name, log_dir=Path(log_dir))
