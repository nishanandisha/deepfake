"""Pre-populates the preprocessing cache for every split.

Run once after building splits and before training. This front-loads the
expensive work (face detection + librosa.pyin) so training epochs become
I/O-bound instead of CPU-bound. Safe to interrupt and re-run -- completed
clips are skipped.
"""

import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.cache import PreprocessingCache, warm_cache
from src.utils.logging import get_logger
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    logger = get_logger("warm_cache", log_dir=Path("outputs") / "warm_cache")

    splits_dir = Path(cfg.data.splits_dir)
    split_files = sorted(splits_dir.glob("*.csv"))
    split_files = [p for p in split_files if p.stem != "dfdc_holdout"]

    if not split_files:
        logger.info(f"No split CSVs in {splits_dir}. Run scripts/build_splits.py first.")
        return

    cache = PreprocessingCache(cfg.data.cache_dir, enabled=True)
    n_mfcc = cfg.model.acoustic.n_mfcc if "acoustic" in cfg.model else cfg.model.n_mfcc

    total_failures = []
    for split_path in split_files:
        manifest = pd.read_csv(split_path)
        logger.info(f"Warming {split_path.stem}: {len(manifest)} clips")

        result = warm_cache(
            manifest, cache,
            frame_rate=cfg.data.frame_rate,
            frame_size=cfg.data.frame_size,
            sample_rate=cfg.data.audio_sample_rate,
            frame_ms=cfg.data.audio_frame_ms,
            hop_ms=cfg.data.audio_hop_ms,
            n_mfcc=n_mfcc,
            pitch_tracker=cfg.data.get("pitch_tracker", "yin"),
            logger=logger,
        )
        total_failures.extend(result["failures"])

    stats = cache.stats()
    logger.info(f"Cache: {stats}")
    if total_failures:
        logger.info(f"{len(total_failures)} clips failed; first few: {total_failures[:5]}")
        failures_path = Path(cfg.data.cache_dir) / "failures.csv"
        pd.DataFrame(total_failures).to_csv(failures_path, index=False)
        logger.info(f"Full failure list: {failures_path}")


if __name__ == "__main__":
    main()
