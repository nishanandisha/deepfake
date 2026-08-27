"""Train the classifier head over cached WavLM embeddings.

Everything expensive already happened in `cache_embeddings.py`, so this loop
only ever touches ~0.4M parameters reading precomputed arrays. That is what
makes an epoch take seconds and hyperparameter changes cheap to test.

Model selection is on val **EER**, not val loss or accuracy. Loss keeps
improving while the model sharpens scores it already ranks correctly, and
accuracy at a fixed 0.5 rewards whichever threshold the class balance
happens to favour. EER is what the project is judged on, so it is what early
stopping watches.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.head import VoiceClassifierHead
from src.preprocessing.dataset import EmbeddingDataset, collate_padded
from src.preprocessing.embeddings import EmbeddingCache, pick_device
from src.training.metrics import compute_metrics


@dataclass
class TrainConfig:
    manifest: str = "data/splits/manifest.csv"
    cache_dir: str = "data/cache"
    layer: int = 6
    out_dir: str = "outputs/run"

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_frames: int = 400
    patience: int = 12
    seed: int = 42

    train_variants: List[str] = field(default_factory=lambda: ["clean", "aug1", "aug2"])
    proj_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.3


def _loader(dataset, batch_size, shuffle=False, sampler=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_padded,
        num_workers=0,  # arrays are already in RAM-friendly .npy; workers add only overhead
    )


def _balanced_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    """Sample classes evenly.

    The corpus is 50/50 overall but the *splits* are not: holding whole
    recording groups together leaves train at roughly 654 human / 524 AI.
    Without this the head drifts toward the majority class and the threshold
    ends up somewhere the calibration split then has to undo.
    """
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = (1.0 / np.maximum(counts, 1))[labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    scores, targets = [], []
    for features, mask, labels in loader:
        logits = model(features.to(device), mask.to(device))
        scores.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.numpy())
    return compute_metrics(np.concatenate(targets), np.concatenate(scores))


@torch.no_grad()
def collect_logits(model, loader, device):
    """Raw logits, for temperature fitting on the calibration split."""
    model.eval()
    out, targets = [], []
    for features, mask, labels in loader:
        out.append(model(features.to(device), mask.to(device)).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(out), np.concatenate(targets)


def train(cfg: TrainConfig) -> Dict[str, float]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(cfg.manifest)
    cache = EmbeddingCache(cfg.cache_dir, cfg.layer)

    train_set = EmbeddingDataset(
        manifest[manifest["split"] == "train"], cache,
        variants=cfg.train_variants, max_frames=cfg.max_frames,
        random_crop=True, spec_augment=True, seed=cfg.seed,
    )
    val_set = EmbeddingDataset(
        manifest[manifest["split"] == "val"], cache,
        variants=["clean"], max_frames=cfg.max_frames,
    )

    train_loader = _loader(
        train_set, cfg.batch_size,
        sampler=_balanced_sampler(train_set.labels(), cfg.seed),
    )
    val_loader = _loader(val_set, cfg.batch_size)

    model = VoiceClassifierHead(
        input_dim=768, proj_dim=cfg.proj_dim,
        hidden_dim=cfg.hidden_dim, dropout=cfg.dropout,
    ).to(device)

    print(f"device={device}  head params={model.num_parameters():,}")
    print(f"train={len(train_set)} (from {len(manifest[manifest['split']=='train'])} clips "
          f"x {len(cfg.train_variants)} variants)  val={len(val_set)}")

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best = {"eer": float("inf"), "epoch": -1}
    history = []
    since_improved = 0
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for features, mask, labels in train_loader:
            optimiser.zero_grad(set_to_none=True)
            logits = model(features.to(device), mask.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            losses.append(loss.item())
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_eer": metrics["eer"],
            "val_auc": metrics["auc"],
            "val_accuracy": metrics["accuracy"],
        }
        history.append(record)

        marker = ""
        if metrics["eer"] < best["eer"]:
            best = {"eer": metrics["eer"], "epoch": epoch, **metrics}
            since_improved = 0
            torch.save(
                {"model": model.state_dict(), "config": cfg.__dict__, "metrics": metrics},
                out_dir / "best.pt",
            )
            marker = "  *"
        else:
            since_improved += 1

        print(
            f"epoch {epoch:3d}  loss {record['train_loss']:.4f}  "
            f"val EER {metrics['eer']*100:6.2f}%  AUC {metrics['auc']:.4f}"
            f"  acc {metrics['accuracy']:.4f}{marker}",
            flush=True,
        )

        if since_improved >= cfg.patience:
            print(f"early stop: no val EER improvement in {cfg.patience} epochs")
            break

    elapsed = time.time() - started
    print(f"\nbest val EER {best['eer']*100:.2f}% at epoch {best['epoch']}  "
          f"({elapsed:.0f}s total)")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "best_metrics.json").write_text(json.dumps(best, indent=2))
    return best
