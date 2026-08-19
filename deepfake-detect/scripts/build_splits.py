"""Index FakeAVCeleb + DFDC and write identity-disjoint split manifests to
data/splits/*.csv. DFDC is indexed only, never split -- see Stage 1 of
deepfake-detection-build-plan.md.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.lavdf import filter_to_existing_files, index_lavdf
from src.preprocessing.manifest import index_dfdc, index_fakeavceleb
from src.preprocessing.splits import (
    assert_no_identity_leakage,
    make_identity_disjoint_splits,
    split_stats,
)
from src.preprocessing.subset import (
    assert_both_classes_present,
    subset_manifest,
    subset_stats,
    warn_if_too_few_real,
)
from src.utils.logging import get_logger
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    logger = get_logger("build_splits", log_dir=Path("outputs") / "build_splits")

    dataset_name = cfg.data.get("name", "lavdf")
    logger.info(f"Indexing {dataset_name} from {cfg.data.root_dir}")

    if dataset_name == "lavdf":
        manifest = index_lavdf(cfg.data.root_dir)
        before = len(manifest)
        manifest = filter_to_existing_files(manifest)
        if len(manifest) < before:
            logger.info(
                f"{before - len(manifest)} of {before} clips listed in metadata are missing "
                "on disk (partial download?) -- continuing with the rest."
            )
    else:
        manifest = index_fakeavceleb(
            cfg.data.root_dir, identity_dir_depth=cfg.data.identity_dir_depth
        )

    if manifest.empty:
        logger.info(
            f"No samples found -- confirm cfg.data.root_dir points at a downloaded "
            f"{dataset_name} release. Run scripts/inspect_dataset.py to check the schema."
        )
        return

    logger.info(f"Indexed {len(manifest)} clips, {manifest['identity_id'].nunique()} identities")

    subset_size = cfg.data.get("subset_size")
    if subset_size:
        manifest = subset_manifest(
            manifest,
            target_size=int(subset_size),
            target_real_fraction=cfg.data.get("subset_real_fraction", 0.25),
            seed=cfg.seed,
            keep_all_real=cfg.data.get("subset_keep_all_real", True),
        )
        stats = subset_stats(manifest)
        logger.info(f"Subset: {stats}")
        assert_both_classes_present(stats)
        warning = warn_if_too_few_real(stats)
        if warning:
            logger.info(f"WARNING: {warning}")

    splits = make_identity_disjoint_splits(
        manifest, split_ratios=dict(cfg.data.split_ratios), seed=cfg.seed
    )
    assert_no_identity_leakage(splits)

    splits_dir = Path(cfg.data.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df in splits.items():
        out_path = splits_dir / f"{split_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Wrote {len(df)} samples to {out_path}")

    logger.info(f"Split stats:\n{split_stats(splits).to_string(index=False)}")

    logger.info(f"Indexing DFDC (held out, no splitting) from {cfg.data.dfdc_root_dir}")
    dfdc_manifest = index_dfdc(cfg.data.dfdc_root_dir)
    dfdc_out_path = splits_dir / "dfdc_holdout.csv"
    dfdc_manifest.to_csv(dfdc_out_path, index=False)
    logger.info(
        f"Wrote {len(dfdc_manifest)} DFDC samples to {dfdc_out_path} (untouched by training)"
    )


if __name__ == "__main__":
    main()
