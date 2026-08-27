"""Training engine: device resolution, the train/eval epoch loop, and the
cosine-with-warmup LR schedule.

The dataset yields (inputs, padding_mask, label) batches and the model is
called as model(inputs, padding_mask=padding_mask), so this loop is
model-agnostic -- it is carried over unchanged from the parent multimodal
project, where the same code trained the visual branch too.
"""

import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.results import write_results_markdown
from src.preprocessing.sampler import class_balanced_loss_weights, make_weighted_sampler
from src.training.checkpointing import TrainingState, load_resumable, save_resumable
from src.training.metrics import compute_binary_classification_metrics
from src.utils.logging import ExperimentLogger


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def make_cosine_warmup_lr_lambda(warmup_epochs: int, max_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return lr_lambda


def make_pos_weight(cfg, train_labels, device: torch.device) -> torch.Tensor:
    """BCE `pos_weight` for the configured imbalance strategy.

    Sampling and loss weighting are two ways to fix the SAME imbalance, so
    applying both double-corrects. That is not hypothetical: with the
    weighted sampler already rebalancing 357 real / 1039 fake to ~50:50,
    the extra pos_weight of 0.344 pushed the model into predicting
    *every* validation clip "real" -- macro-F1 pinned at exactly 0.1903
    while AUC still rose, because ranking was fine and only the operating
    point was wrecked.

    So: the sampler takes precedence, and loss weighting only applies when
    sampling is off (or when `use_loss_weighting` is set explicitly).
    """
    sampler_on = cfg.data.get("use_class_balanced_sampler", True)
    loss_weighting = cfg.data.get("use_loss_weighting", None)
    if loss_weighting is None:
        loss_weighting = not sampler_on  # don't stack the two corrections

    if not loss_weighting:
        return torch.tensor(1.0, device=device)

    class_weights = class_balanced_loss_weights(train_labels, enabled=True)
    return (class_weights[1] / class_weights[0]).to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer=None,
    grad_clip_norm: float = None,
) -> tuple:
    """One pass over `loader`. Trains if `optimizer` is given, else eval-only
    (no_grad). Returns (avg_loss, y_true list, y_prob list)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, num_samples = 0.0, 0
    y_true, y_prob = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, padding_mask, labels in loader:
            inputs, padding_mask, labels = (
                inputs.to(device),
                padding_mask.to(device),
                labels.to(device),
            )

            logits = model(inputs, padding_mask=padding_mask)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            num_samples += len(labels)
            y_true.extend(labels.detach().cpu().tolist())
            y_prob.extend(torch.sigmoid(logits).detach().cpu().tolist())

    return total_loss / max(num_samples, 1), y_true, y_prob


def run_standalone_training(
    cfg,
    logger: ExperimentLogger,
    model: nn.Module,
    train_dataset,
    val_dataset,
    branch_name: str,
) -> Dict[str, float]:
    """The training loop: sampler/loader setup, class-weighted BCE, AdamW +
    cosine-warmup schedule, early stopping on val AUC, checkpointing, and
    results-markdown writing.
    """
    device = resolve_device(cfg.device)
    logger.info(f"Training {branch_name} on device={device}")
    model = model.to(device)

    train_labels = train_dataset.df["label"].tolist()
    sampler = make_weighted_sampler(train_labels, enabled=cfg.data.use_class_balanced_sampler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=4 if cfg.data.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=4 if cfg.data.num_workers > 0 else None,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=make_pos_weight(cfg, train_labels, device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        make_cosine_warmup_lr_lambda(cfg.training.warmup_epochs, cfg.training.max_epochs),
    )

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    resume_path = checkpoint_dir / "last.pt"

    state = TrainingState()
    if cfg.training.get("resume", True):
        restored = load_resumable(str(resume_path), model, optimizer, scheduler, device)
        if restored is not None:
            state = restored
            logger.info(
                f"Resumed from {resume_path} at epoch {state.epoch} "
                f"(best_auc={state.best_auc:.4f}) -- delete last.pt to start fresh"
            )

    best_auc = state.best_auc
    best_val_loss = state.best_val_loss
    epochs_without_improvement = state.epochs_without_improvement
    best_val_metrics: Dict[str, float] = dict(state.best_val_metrics)
    start_epoch = state.epoch

    if start_epoch >= cfg.training.max_epochs:
        logger.info(
            f"Already trained {start_epoch}/{cfg.training.max_epochs} epochs; nothing to do. "
            "Raise training.max_epochs to continue."
        )
        return best_val_metrics

    for epoch in range(start_epoch, cfg.training.max_epochs):
        train_loss, _, _ = run_epoch(
            model, train_loader, device, criterion, optimizer, cfg.training.grad_clip_norm
        )
        scheduler.step()

        val_loss, val_y_true, val_y_prob = run_epoch(model, val_loader, device, criterion)
        val_metrics = compute_binary_classification_metrics(val_y_true, val_y_prob)

        logger.info(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        scalar_log = {"train/loss": train_loss, "val/loss": val_loss}
        scalar_log.update({f"val/{k}": v for k, v in val_metrics.items()})
        logger.scalars(scalar_log, step=epoch)

        current_auc = val_metrics["auc"]
        # AUC is scale/threshold-invariant, so it can plateau (e.g. at 1.0)
        # while the model keeps getting more confident/better-calibrated on
        # ties -- break ties on val_loss so checkpointing doesn't freeze on
        # the first, least-trained epoch that happened to hit the plateau.
        is_improvement = current_auc == current_auc and (
            current_auc > best_auc or (current_auc == best_auc and val_loss < best_val_loss)
        )
        if is_improvement:
            best_auc = current_auc
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        # Written every epoch (not just on improvement) so an interrupted
        # run resumes from where it stopped, not from the last best epoch.
        save_resumable(
            str(resume_path), model, optimizer, scheduler,
            TrainingState(
                epoch=epoch + 1,
                best_auc=best_auc,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                best_val_metrics=best_val_metrics,
            ),
        )

        if epochs_without_improvement >= cfg.training.early_stopping_patience:
            logger.info(
                f"Early stopping at epoch {epoch} (no val AUC improvement for "
                f"{cfg.training.early_stopping_patience} epochs)"
            )
            break

    write_results_markdown(
        output_path=str(Path(cfg.output_dir) / "results.md"),
        branch_name=branch_name,
        metrics=best_val_metrics,
        data_source_note=f"val split from {cfg.data.splits_dir}",
    )
    logger.info(f"Best val metrics: {best_val_metrics}")
    return best_val_metrics
