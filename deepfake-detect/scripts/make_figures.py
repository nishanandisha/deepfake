"""Generates the results-section figures for the paper from
outputs_medium/test_raw_arrays.npz (raw per-clip probs + gate values,
produced by evaluate_test.py) and calibration/policy.json (tau thresholds).

Run after evaluate_test.py has written test_raw_arrays.npz:
  python scripts/make_figures.py --outputs D:/deepfake-data/outputs_medium
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                              roc_curve)

# Reference palette (references/palette.md) -- light-mode categorical slots.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

MODEL_COLORS = {
    "visual": BLUE,
    "acoustic": ORANGE,
    "late_fusion": AQUA,
    "cross_attention": VIOLET,
}
MODEL_LABELS = {
    "visual": "Visual CNN + Transformer",
    "acoustic": "Audio Transformer",
    "late_fusion": "Late fusion (avg.)",
    "cross_attention": "Proposed (cross-attention)",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "font.size": 11,
})


def fig_roc(labels, probs, dest):
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)
    for key in ["visual", "acoustic", "late_fusion", "cross_attention"]:
        fpr, tpr, _ = roc_curve(labels, probs[key])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=MODEL_COLORS[key], linewidth=2,
                label=f"{MODEL_LABELS[key]} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Split")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def fig_pr(labels, probs, dest):
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)
    base_rate = labels.mean()
    for key in ["visual", "acoustic", "late_fusion", "cross_attention"]:
        prec, rec, _ = precision_recall_curve(labels, probs[key])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=MODEL_COLORS[key], linewidth=2,
                label=f"{MODEL_LABELS[key]} (AP={pr_auc:.3f})")
    ax.axhline(base_rate, color=MUTED, linewidth=1, linestyle="--",
               label=f"Base rate ({base_rate:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves — Test Split")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(labels, probs, dest, key="cross_attention", threshold=0.5):
    pred = (probs[key] >= threshold).astype(int)
    cm = confusion_matrix(labels, pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Authentic", "Manipulated"])
    ax.set_yticklabels(["Authentic", "Manipulated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Confusion Matrix — {MODEL_LABELS[key]}")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] > 50 else INK
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row %")
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def fig_score_distribution(labels, probs, tau_lo, tau_hi, dest, key="cross_attention"):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    bins = np.linspace(0, 1, 41)
    ax.hist(probs[key][labels == 0], bins=bins, alpha=0.75, color=BLUE,
            label="Authentic", edgecolor="white", linewidth=0.3)
    ax.hist(probs[key][labels == 1], bins=bins, alpha=0.75, color=ORANGE,
            label="Manipulated", edgecolor="white", linewidth=0.3)
    ax.axvline(tau_lo, color=GOOD, linewidth=1.5, linestyle="--",
               label=f"$\\tau_{{lo}}$={tau_lo:.3f}")
    ax.axvline(tau_hi, color=CRITICAL, linewidth=1.5, linestyle="--",
               label=f"$\\tau_{{hi}}$={tau_hi:.2f}")
    ax.set_xlabel("Calibrated authenticity score")
    ax.set_ylabel("Clip count")
    ax.set_title("Score Distribution by Ground Truth — Test Split")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def fig_gate_distribution(gate, dest):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.hist(gate, bins=40, color=VIOLET, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.mean(gate)), color=INK, linewidth=1.5, linestyle="--",
               label=f"mean={np.mean(gate):.3f}")
    ax.set_xlabel("Learned modality gate value (0 = acoustic, 1 = visual)")
    ax.set_ylabel("Clip count")
    ax.set_title("Cross-Modal Gate Distribution — Test Split")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def fig_auc_bar(test_results_json, dest):
    payload = json.loads(Path(test_results_json).read_text())
    order = ["visual", "acoustic", "late_fusion", "cross_attention"]
    aucs = [payload["models"][k]["auc"] for k in order]
    lo = [payload["models"][k]["auc"] - payload["models"][k]["auc_ci_low"] for k in order]
    hi = [payload["models"][k]["auc_ci_high"] - payload["models"][k]["auc"] for k in order]
    colors = [MODEL_COLORS[k] for k in order]
    labels_short = ["Visual", "Acoustic", "Late fusion", "Proposed\n(cross-attn.)"]

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    x = np.arange(len(order))
    ax.bar(x, aucs, yerr=[lo, hi], color=colors, capsize=5, width=0.6,
           error_kw={"linewidth": 1.3, "ecolor": INK})
    ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=9.5)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.9, 1.015)
    ax.set_title("Model AUC with 95% Bootstrap CI — Test Split")
    for xi, v, h in zip(x, aucs, hi):
        ax.text(xi, v + h + 0.004, f"{v:.3f}", ha="center", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="D:/deepfake-data/outputs_medium")
    args = parser.parse_args()

    out = Path(args.outputs)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(out / "test_raw_arrays.npz")
    labels = raw["labels"]
    probs = {k: raw[k] for k in ["visual", "acoustic", "late_fusion", "cross_attention"]}
    gate = raw["gate"]

    policy = json.loads((out / "calibration" / "policy.json").read_text())
    tau_lo, tau_hi = policy["tau_lo"], policy["tau_hi"]

    fig_roc(labels, probs, fig_dir / "fig1_roc_curves.png")
    fig_pr(labels, probs, fig_dir / "fig2_pr_curves.png")
    fig_confusion(labels, probs, fig_dir / "fig3_confusion_matrix.png")
    fig_score_distribution(labels, probs, tau_lo, tau_hi, fig_dir / "fig4_score_distribution.png")
    fig_gate_distribution(gate, fig_dir / "fig5_gate_distribution.png")
    fig_auc_bar(out / "test_results.json", fig_dir / "fig6_auc_comparison.png")

    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
