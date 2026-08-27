"""Results-markdown writer, so every training and evaluation run produces a
consistently formatted summary. Also writes a sibling results.json with the
raw metrics dict, so later steps (scripts/export_model.py embeds it as the
artefact's test metrics) can load the numbers programmatically instead of
parsing markdown."""

import json
from pathlib import Path
from typing import Dict, Optional

# Stop and debug if AUC is near chance. 0.55 was too lenient: a branch
# scoring AUC 0.599 while predicting a single class for every clip still
# printed "OK". An AUC in the 0.6s on a binary task is barely-above-chance
# ranking, not a working detector, so the bar is 0.65 and degenerate
# predictors are called out separately regardless of AUC.
NEAR_CHANCE_AUC_THRESHOLD = 0.65


def _degenerate_predictor_note(metrics: Dict[str, float]) -> Optional[str]:
    """Flags a classifier that is really just predicting one class.

    Accuracy alone hides this completely: on a 76% fake split, always
    answering "fake" scores 0.765 accuracy and 1.0 recall while having
    learned nothing at all.
    """
    recall = metrics.get("recall")
    precision = metrics.get("precision")
    if recall is None or precision is None:
        return None

    if recall >= 0.999:
        return (
            "**Degenerate predictor:** recall is 1.0, i.e. every sample was labelled "
            f"positive. Accuracy ({metrics.get('accuracy', float('nan')):.4f}) merely "
            "reflects the class prior here, not detection skill."
        )
    if recall <= 0.001:
        return (
            "**Degenerate predictor:** recall is 0.0, i.e. every sample was labelled "
            "negative. The model has not learned to separate the classes."
        )
    return None


def load_metrics_json(path: str) -> Optional[Dict[str, float]]:
    """Reads a results.json written by write_results_markdown, or None if
    it doesn't exist yet (e.g. that stage hasn't been run)."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_results_markdown(
    output_path: str,
    branch_name: str,
    metrics: Dict[str, float],
    data_source_note: str,
    extra_notes: Optional[str] = None,
) -> None:
    auc = metrics.get("auc", float("nan"))
    if auc != auc:  # NaN check without importing math/numpy here
        status = "UNKNOWN (AUC undefined -- only one class present in eval set)"
    elif auc < NEAR_CHANCE_AUC_THRESHOLD:
        status = (
            f"NEAR-CHANCE (AUC={auc:.3f} < {NEAR_CHANCE_AUC_THRESHOLD}) -- "
            "STOP AND DEBUG DATA/LABELS BEFORE PROCEEDING."
        )
    else:
        status = f"OK (AUC={auc:.3f})"

    lines = [
        f"# {branch_name} -- standalone results",
        "",
        f"**Status:** {status}",
        "",
    ]

    degenerate = _degenerate_predictor_note(metrics)
    if degenerate:
        lines += [degenerate, ""]

    lines += [
        f"**Data source:** {data_source_note}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    ordered_keys = ["accuracy", "precision", "recall", "macro_f1", "auc", "eer"]
    extra_keys = [k for k in metrics if k not in ordered_keys]
    for key in ordered_keys + extra_keys:
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]:.4f} |")

    if extra_notes:
        lines += ["", extra_notes]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")
    Path(output_path).with_suffix(".json").write_text(json.dumps(metrics, indent=2))
