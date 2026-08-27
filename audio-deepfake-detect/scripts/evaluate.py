"""Evaluates the trained acoustic model on the held-out TEST split.

Separate from the training loop on purpose: that loop reports validation
metrics, and validation was also used for early stopping, so those numbers
are optimistically biased. Every headline figure should come from the test
split, computed once.

Reports accuracy / precision / recall / macro-F1 / AUC / EER plus a
bootstrap 95% CI on the AUC, so the number carries an honest error bar
instead of being a bare point estimate.

Usage:
  python scripts/evaluate.py                       # test split, default config
  python scripts/evaluate.py data=lean split=val
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.datasets import build_split_dataset
from src.evaluation.results import write_results_markdown
from src.inference.loader import load_acoustic_model
from src.training.common import resolve_device
from src.training.metrics import compute_binary_classification_metrics
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def auc_ci(labels: np.ndarray, probs: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Bootstrap 95% CI for the AUC."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    scores = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(labels[pick])) < 2:
            continue
        scores.append(roc_auc_score(labels[pick], probs[pick]))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    logger = get_logger("evaluate", log_dir=Path(cfg.output_dir))
    device = resolve_device(cfg.device)

    split = cfg.get("split", "test")
    loaded = load_acoustic_model(cfg.model_artefact, device)
    logger.info(f"Loaded {cfg.model_artefact} ({loaded.input_dim} input features)")

    dataset = build_split_dataset(cfg, split, loaded.n_mfcc)
    logger.info(f"Evaluating on {len(dataset)} {split} samples")

    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    y_true, y_prob = [], []
    with torch.no_grad():
        for features, mask, labels in loader:
            logits = loaded.model(features.to(device), padding_mask=mask.to(device))
            y_prob.extend(torch.sigmoid(logits).cpu().tolist())
            y_true.extend(labels.tolist())

    metrics = compute_binary_classification_metrics(y_true, y_prob)
    low, high = auc_ci(np.array(y_true), np.array(y_prob), seed=cfg.seed)
    metrics["auc_ci_low"], metrics["auc_ci_high"] = low, high

    logger.info(json.dumps(metrics, indent=2))

    out_path = Path(cfg.output_dir) / f"{split}_results.md"
    write_results_markdown(
        output_path=str(out_path),
        branch_name=f"Audio Transformer -- {split} split",
        metrics=metrics,
        data_source_note=f"{split} split from {cfg.data.splits_dir} "
                         f"({len(dataset)} clips), model {cfg.model_artefact}",
    )
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
