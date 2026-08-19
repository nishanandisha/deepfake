"""Joint cross-modal fusion training loop (Stage 5 of the build plan, the
core contribution). Unlike the standalone branches, batches carry both
modalities at once, and the loss combines the fused prediction with each
branch's own auxiliary prediction -- so this doesn't reuse
run_standalone_training, but mirrors its structure closely.
"""

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.results import load_metrics_json, write_results_markdown
from src.models.acoustic.encoder import AcousticEncoder
from src.models.fusion.cross_attention import CrossModalFusion, load_standalone_checkpoint_into
from src.models.visual.encoder import VisualEncoder
from src.preprocessing.cache import get_shared_cache
from src.preprocessing.dataset import MultimodalDataset
from src.preprocessing.sampler import make_weighted_sampler
from src.training.checkpointing import TrainingState, load_resumable, save_resumable
from src.training.common import (
    make_cosine_warmup_lr_lambda,
    make_pos_weight,
    resolve_device,
)
from src.training.metrics import compute_binary_classification_metrics
from src.training.train_acoustic import acoustic_input_dim
from src.utils.logging import ExperimentLogger, get_logger


def build_fusion_model(cfg) -> CrossModalFusion:
    visual_cfg, acoustic_cfg = cfg.model.visual, cfg.model.acoustic
    assert visual_cfg.embed_dim == acoustic_cfg.embed_dim, (
        "visual/acoustic embed_dim must match for cross-attention to combine Hv and Ha"
    )

    visual_encoder = VisualEncoder(
        backbone=visual_cfg.backbone,
        pretrained=visual_cfg.pretrained,
        embed_dim=visual_cfg.embed_dim,
        transformer_depth=visual_cfg.transformer.depth,
        transformer_heads=visual_cfg.transformer.heads,
        transformer_ff_dim=visual_cfg.transformer.ff_dim,
        dropout=visual_cfg.transformer.dropout,
    )
    acoustic_encoder = AcousticEncoder(
        input_dim=acoustic_input_dim(acoustic_cfg.n_mfcc),
        embed_dim=acoustic_cfg.embed_dim,
        transformer_depth=acoustic_cfg.transformer.depth,
        transformer_heads=acoustic_cfg.transformer.heads,
        transformer_ff_dim=acoustic_cfg.transformer.ff_dim,
        dropout=acoustic_cfg.transformer.dropout,
    )

    model = CrossModalFusion(
        visual_encoder,
        acoustic_encoder,
        embed_dim=visual_cfg.embed_dim,
        cross_attention_heads=cfg.model.cross_attention.heads,
        cross_attention_dropout=cfg.model.cross_attention.dropout,
        gate_hidden_dim=cfg.model.gate.hidden_dim,
        pooling=visual_cfg.get("pooling", "mean"),
    )

    if cfg.get("init_from_standalone_checkpoints", False):
        load_standalone_checkpoint_into(
            model.visual_encoder, model.visual_aux_head, cfg.visual_checkpoint
        )
        load_standalone_checkpoint_into(
            model.acoustic_encoder, model.acoustic_aux_head, cfg.acoustic_checkpoint
        )

    return model


def build_fusion_model_for_inference(cfg) -> CrossModalFusion:
    """Builds the fusion model with warm-starting disabled.

    build_fusion_model() warm-starts each branch from its Stage 2/3
    checkpoint when init_from_standalone_checkpoints is set (the training
    default). When you are about to load a *trained fusion* checkpoint that
    is both pointless -- every weight gets overwritten -- and fragile,
    because those training-time paths need not exist wherever the model is
    being loaded. Any code path that loads fusion/best.pt should use this.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.init_from_standalone_checkpoints = False
    return build_fusion_model(cfg)


def run_fusion_epoch(
    model: CrossModalFusion,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    lambda_visual: float,
    lambda_acoustic: float,
    optimizer=None,
    grad_clip_norm: float = None,
) -> tuple:
    """Returns (avg_loss, y_true, y_prob, avg_gate). y_true/y_prob come from
    the fused prediction only; avg_gate is the mean visual-vs-acoustic gate
    weighting over the epoch (sanity-check against modality collapse)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, num_samples = 0.0, 0
    y_true, y_prob, gate_values = [], [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for frames, v_mask, features, a_mask, labels in loader:
            frames, v_mask = frames.to(device), v_mask.to(device)
            features, a_mask = features.to(device), a_mask.to(device)
            labels = labels.to(device)

            outputs = model(frames, v_mask, features, a_mask)
            loss = (
                criterion(outputs["y_hat_logit"], labels)
                + lambda_visual * criterion(outputs["y_hat_visual_logit"], labels)
                + lambda_acoustic * criterion(outputs["y_hat_acoustic_logit"], labels)
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            num_samples += len(labels)
            y_true.extend(labels.detach().cpu().tolist())
            y_prob.extend(torch.sigmoid(outputs["y_hat_logit"]).detach().cpu().tolist())
            gate_values.extend(outputs["gate"].detach().cpu().tolist())

    avg_gate = sum(gate_values) / max(len(gate_values), 1)
    return total_loss / max(num_samples, 1), y_true, y_prob, avg_gate


def _compare_with_late_fusion(
    logger: ExperimentLogger, late_fusion_path: str, fusion_metrics: Dict
) -> str:
    late_fusion_metrics = load_metrics_json(late_fusion_path)
    if late_fusion_metrics is None:
        return (
            "**Comparison with late fusion:** not available -- run Stage 4 "
            f"(scripts/evaluate_late_fusion.py) first so {late_fusion_path} exists."
        )

    fusion_auc = fusion_metrics.get("auc", float("nan"))
    late_auc = late_fusion_metrics.get("auc", float("nan"))
    fusion_f1 = fusion_metrics.get("macro_f1", float("nan"))
    late_f1 = late_fusion_metrics.get("macro_f1", float("nan"))
    auc_delta = fusion_auc - late_auc
    f1_delta = fusion_f1 - late_f1
    verdict = "BEATS" if auc_delta > 0 and f1_delta > 0 else "DOES NOT CLEARLY BEAT"

    note = (
        f"**Comparison with late fusion:** cross-attention {verdict} late fusion on this "
        f"val split (AUC {fusion_auc:.4f} vs {late_auc:.4f}, delta={auc_delta:+.4f}; "
        f"macro-F1 {fusion_f1:.4f} vs {late_f1:.4f}, delta={f1_delta:+.4f}). "
        "This is the build plan's central empirical claim -- see Stage 5 exit criteria."
    )
    logger.info(note)
    return note


def train_fusion(cfg, logger: ExperimentLogger = None) -> Dict[str, float]:
    logger = logger or get_logger("fusion", log_dir=Path(cfg.output_dir))
    device = resolve_device(cfg.device)
    logger.info(f"Training cross-modal fusion on device={device}")

    splits_dir = Path(cfg.data.splits_dir)
    visual_kwargs = dict(
        frame_rate=cfg.data.frame_rate,
        frame_size=cfg.data.frame_size,
        num_frames=cfg.data.get("num_frames", 32),
    )
    acoustic_kwargs = dict(
        sample_rate=cfg.data.audio_sample_rate,
        frame_ms=cfg.data.audio_frame_ms,
        hop_ms=cfg.data.audio_hop_ms,
        n_mfcc=cfg.model.acoustic.n_mfcc,
        pitch_tracker=cfg.data.get("pitch_tracker", "yin"),
        num_frames=cfg.data.get("num_audio_frames", 300),
    )

    cache = get_shared_cache(cfg.data.get("cache_dir"), cfg.data.get("use_cache", True))
    train_dataset = MultimodalDataset(
        splits_dir / "train.csv", split="train",
        visual_kwargs=visual_kwargs, acoustic_kwargs=acoustic_kwargs, seed=cfg.seed, cache=cache,
    )
    val_dataset = MultimodalDataset(
        splits_dir / "val.csv", split="val",
        visual_kwargs=visual_kwargs, acoustic_kwargs=acoustic_kwargs, seed=cfg.seed, cache=cache,
    )

    train_labels = train_dataset.df["label"].tolist()
    sampler = make_weighted_sampler(train_labels, enabled=cfg.data.use_class_balanced_sampler)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=4 if cfg.data.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=4 if cfg.data.num_workers > 0 else None,
    )

    model = build_fusion_model(cfg).to(device)

    # See make_pos_weight: sampler and loss weighting must not both apply.
    criterion = nn.BCEWithLogitsLoss(pos_weight=make_pos_weight(cfg, train_labels, device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_cosine_warmup_lr_lambda(cfg.training.warmup_epochs, cfg.training.max_epochs)
    )

    lambda_visual = cfg.model.aux_loss.lambda_visual
    lambda_acoustic = cfg.model.aux_loss.lambda_acoustic

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

    if state.epoch >= cfg.training.max_epochs:
        logger.info(
            f"Already trained {state.epoch}/{cfg.training.max_epochs} epochs; nothing to do."
        )
        return best_val_metrics

    for epoch in range(state.epoch, cfg.training.max_epochs):
        train_loss, _, _, train_gate = run_fusion_epoch(
            model, train_loader, device, criterion, lambda_visual, lambda_acoustic,
            optimizer, cfg.training.grad_clip_norm,
        )
        scheduler.step()

        val_loss, val_y_true, val_y_prob, val_gate = run_fusion_epoch(
            model, val_loader, device, criterion, lambda_visual, lambda_acoustic
        )
        val_metrics = compute_binary_classification_metrics(val_y_true, val_y_prob)
        val_metrics["avg_gate_visual_weight"] = val_gate

        logger.info(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"train_gate={train_gate:.3f} val_gate={val_gate:.3f}"
        )
        scalar_log = {"train/loss": train_loss, "val/loss": val_loss, "train/gate": train_gate}
        scalar_log.update({f"val/{k}": v for k, v in val_metrics.items()})
        logger.scalars(scalar_log, step=epoch)

        current_auc = val_metrics["auc"]
        # See src/training/common.py::run_standalone_training for why ties
        # on AUC are broken by val_loss instead of freezing the checkpoint
        # at the first epoch that hit a plateau.
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

    comparison_note = _compare_with_late_fusion(logger, cfg.late_fusion_results, best_val_metrics)

    write_results_markdown(
        output_path=str(Path(cfg.output_dir) / "results.md"),
        branch_name="Proposed (cross-attention fusion)",
        metrics=best_val_metrics,
        data_source_note=f"val split from {splits_dir}",
        extra_notes=comparison_note,
    )
    logger.info(f"Best val metrics: {best_val_metrics}")
    return best_val_metrics
