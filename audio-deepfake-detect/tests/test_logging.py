from pathlib import Path

from src.utils.logging import get_logger


def test_get_logger_creates_log_dir(tmp_path: Path):
    log_dir = tmp_path / "run1"
    logger = get_logger("test_run", log_dir=log_dir)

    logger.info("hello")
    logger.scalar("dummy/metric", 1.0, step=0)
    logger.close()

    assert (log_dir / "run.log").exists()
