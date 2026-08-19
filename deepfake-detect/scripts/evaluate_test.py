"""Final evaluation on the held-out TEST split, plus a DeLong significance test.

Why this exists separately from the training loops: those report validation
metrics, and validation was also used for early stopping and threshold
selection, so those numbers are optimistically biased. The build plan's
reporting rule is that every headline figure comes from the test split
only. This script produces those figures, once, for all four models.

It also runs DeLong's test on the two fusion AUCs, so the paper can state
whether the cross-attention vs late-fusion difference is significant
rather than leaving the reader to eyeball two close numbers.

Usage:
  python scripts/evaluate_test.py --outputs D:/deepfake-data/outputs_medium \\
                                  --splits  D:/deepfake-data/splits_medium \\
                                  --cache   D:/deepfake-data/cache_medium
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.fusion.late_fusion import (
    average_probabilities,
    load_frozen_checkpoint,
    predict_branch_probabilities,
    resolve_device,
)
from src.preprocessing.cache import get_shared_cache
from src.preprocessing.dataset import AcousticDataset, MultimodalDataset, VisualDataset
from src.training.metrics import compute_binary_classification_metrics
from src.training.train_acoustic import build_acoustic_model
from src.training.train_fusion import build_fusion_model_for_inference
from src.training.train_visual import build_visual_model

# ---------------------------------------------------------------- DeLong

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(labels: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray) -> dict:
    """DeLong's test for two correlated ROC curves on the same samples.

    Returns each AUC, the difference, the z statistic and a two-sided
    p-value. Correlated because both models score the *same* clips -- an
    independent two-sample test would overstate the difference.
    """
    pos = labels == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())

    aucs, v01, v10 = [], [], []
    for probs in (probs_a, probs_b):
        x, y = probs[pos], probs[neg]
        tx, ty, tz = _midrank(x), _midrank(y), _midrank(np.concatenate([x, y]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)

    v01, v10 = np.array(v01), np.array(v10)
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    cov = s01 / m + s10 / n

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": aucs[0], "auc_b": aucs[1], "diff": diff,
            "z": float(z), "p_value": float(p), "var": float(var)}


def auc_ci(labels: np.ndarray, probs: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Bootstrap 95% CI for a single AUC -- gives the table an honest
    error bar instead of a bare point estimate."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    scores = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(labels[pick])) < 2:
            continue
        scores.append(roc_auc_score(labels[pick], probs[pick]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ---------------------------------------------------------------- eval

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all models on the test split.")
    parser.add_argument("--outputs", default="D:/deepfake-data/outputs_medium")
    parser.add_argument("--splits", default="D:/deepfake-data/splits_medium")
    parser.add_argument("--cache", default="D:/deepfake-data/cache_medium")
    parser.add_argument("--data", default="medium")
    args = parser.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["model=fusion", f"data={args.data}"])

    device = resolve_device("auto")
    cache = get_shared_cache(args.cache, True)
    test_csv = Path(args.splits) / "test.csv"

    vkw = dict(frame_rate=cfg.data.frame_rate, frame_size=cfg.data.frame_size,
               num_frames=cfg.data.get("num_frames", 16))
    akw = dict(sample_rate=cfg.data.audio_sample_rate, frame_ms=cfg.data.audio_frame_ms,
               hop_ms=cfg.data.audio_hop_ms, n_mfcc=cfg.model.acoustic.n_mfcc,
               pitch_tracker=cfg.data.get("pitch_tracker", "yin"),
               num_frames=cfg.data.get("num_audio_frames", 400))

    visual_ds = VisualDataset(test_csv, split="test", seed=cfg.seed, cache=cache, **vkw)
    acoustic_ds = AcousticDataset(test_csv, split="test", seed=cfg.seed, cache=cache, **akw)
    labels = np.array([1 if v == "fake" else 0 for v in visual_ds.df["label"]])

    out = Path(args.outputs)
    results, probs = {}, {}

    print(f"Evaluating on TEST split: {len(labels)} clips "
          f"({(labels == 0).sum()} real / {(labels == 1).sum()} fake)\n", flush=True)

    # --- branches -------------------------------------------------------
    vm = load_frozen_checkpoint(build_visual_model(cfg.model.visual),
                                str(out / "visual/checkpoints/best.pt"), device)
    probs["visual"] = np.array(predict_branch_probabilities(
        vm, visual_ds, device, cfg.data.batch_size, 0))
    del vm
    torch.cuda.empty_cache()

    am = load_frozen_checkpoint(build_acoustic_model(cfg.model.acoustic),
                                str(out / "acoustic/checkpoints/best.pt"), device)
    probs["acoustic"] = np.array(predict_branch_probabilities(
        am, acoustic_ds, device, cfg.data.batch_size, 0))
    del am
    torch.cuda.empty_cache()

    probs["late_fusion"] = np.array(average_probabilities(
        list(probs["visual"]), list(probs["acoustic"]), 0.5))

    # --- cross-attention fusion ----------------------------------------
    multi_ds = MultimodalDataset(test_csv, split="test", visual_kwargs=vkw,
                                 acoustic_kwargs=akw, seed=cfg.seed, cache=cache)
    fm = load_frozen_checkpoint(build_fusion_model_for_inference(cfg),
                                str(out / "fusion/checkpoints/best.pt"), device).to(device)
    fused, gates = [], []
    with torch.no_grad():
        for fr, vmk, fe, amk, _ in DataLoader(multi_ds, batch_size=cfg.data.batch_size,
                                              shuffle=False, num_workers=0):
            o = fm(fr.to(device), vmk.to(device), fe.to(device), amk.to(device))
            fused.extend(torch.sigmoid(o["y_hat_logit"]).cpu().tolist())
            gates.extend(o["gate"].detach().cpu().reshape(-1).tolist())
    probs["cross_attention"] = np.array(fused)
    gate_values = np.array(gates)
    del fm
    torch.cuda.empty_cache()

    # --- metrics --------------------------------------------------------
    names = {"visual": "Visual CNN + Transformer", "acoustic": "Audio Transformer",
             "late_fusion": "Late fusion (avg.)", "cross_attention": "Proposed (cross-attention)"}
    print(f"{'Model':30s} {'AUC':>7s} {'95% CI':>16s} {'MacroF1':>8s} "
          f"{'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'EER':>7s}")
    print("-" * 92)
    for key, label in names.items():
        m = compute_binary_classification_metrics(labels, probs[key])
        lo, hi = auc_ci(labels, probs[key])
        results[key] = {**m, "auc_ci_low": lo, "auc_ci_high": hi}
        print(f"{label:30s} {m['auc']:7.4f} [{lo:.4f},{hi:.4f}] {m['macro_f1']:8.4f} "
              f"{m['accuracy']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f} {m['eer']:7.4f}")

    # --- significance ---------------------------------------------------
    print("\nDeLong test - cross-attention vs late fusion (same clips, correlated ROCs)")
    d = delong_test(labels, probs["cross_attention"], probs["late_fusion"])
    print(f"  AUC cross-attention : {d['auc_a']:.4f}")
    print(f"  AUC late fusion     : {d['auc_b']:.4f}")
    print(f"  difference          : {d['diff']:+.4f}")
    print(f"  z = {d['z']:.3f},  p = {d['p_value']:.4f}")
    verdict = ("NOT significant (p > 0.05) -- the two are statistically indistinguishable"
               if d["p_value"] > 0.05 else "SIGNIFICANT (p < 0.05)")
    print(f"  -> {verdict}")

    d2 = delong_test(labels, probs["late_fusion"], probs["visual"])
    print("\nDeLong test - late fusion vs best single branch (visual)")
    print(f"  difference = {d2['diff']:+.4f},  z = {d2['z']:.3f},  p = {d2['p_value']:.4f}")
    sig2 = "NOT significant" if d2["p_value"] > 0.05 else "SIGNIFICANT"
    print(f"  -> {sig2} (p {'>' if d2['p_value'] > 0.05 else '<'} 0.05)")

    payload = {"split": "test", "n_clips": int(len(labels)),
               "n_real": int((labels == 0).sum()), "n_fake": int((labels == 1).sum()),
               "models": results,
               "delong_crossattn_vs_latefusion": d,
               "delong_latefusion_vs_visual": d2}
    dest = out / "test_results.json"
    dest.write_text(json.dumps(payload, indent=2))
    print(f"\nWritten to {dest}")

    # Raw per-clip arrays for figure generation (ROC/PR curves, histograms,
    # confusion matrices) -- the JSON above only has aggregated metrics.
    raw_dest = out / "test_raw_arrays.npz"
    np.savez(raw_dest, labels=labels, gate=gate_values,
              **{k: v for k, v in probs.items()})
    print(f"Raw arrays written to {raw_dest}")


if __name__ == "__main__":
    main()
