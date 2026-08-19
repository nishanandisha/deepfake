"""Stage 7, part 5: attribution agreement rate.

For manipulated clips whose ground-truth `manipulated_modality` is known
(video-only fake, audio-only fake, or both), checks whether the coarse
SHAP modality split points at the modality that was actually manipulated.

This is the evidence that the explanations are *correct*, not merely
present -- without it, "explainable" is an unfalsifiable claim. Real
samples (manipulated_modality == "none") are excluded: there is no
manipulated modality to agree with.
"""

from typing import Dict, List, Sequence

from src.explain.shap_modality import implicated_modality

EXPLAINABLE_MODALITIES = {"video", "audio", "both"}


def compute_attribution_agreement(
    modality_splits: Sequence[Dict[str, float]],
    ground_truth_modalities: Sequence[str],
    dominance_margin: float = 0.15,
) -> Dict[str, object]:
    """Returns the overall agreement rate plus a per-modality breakdown and
    a confusion table of predicted vs. true manipulated modality."""
    assert len(modality_splits) == len(ground_truth_modalities), (
        "modality_splits and ground_truth_modalities must be aligned"
    )

    per_modality_correct: Dict[str, int] = {}
    per_modality_total: Dict[str, int] = {}
    confusion: Dict[str, Dict[str, int]] = {}
    predictions: List[str] = []

    considered = 0
    correct = 0

    for split, truth in zip(modality_splits, ground_truth_modalities):
        if truth not in EXPLAINABLE_MODALITIES:
            predictions.append("")  # keep index alignment with the input
            continue

        predicted = implicated_modality(split, dominance_margin=dominance_margin)
        predictions.append(predicted)

        considered += 1
        per_modality_total[truth] = per_modality_total.get(truth, 0) + 1
        confusion.setdefault(truth, {})
        confusion[truth][predicted] = confusion[truth].get(predicted, 0) + 1

        if predicted == truth:
            correct += 1
            per_modality_correct[truth] = per_modality_correct.get(truth, 0) + 1

    agreement_rate = correct / considered if considered else float("nan")

    return {
        "attribution_agreement_rate": agreement_rate,
        "num_considered": considered,
        "num_correct": correct,
        "per_modality_agreement": {
            modality: per_modality_correct.get(modality, 0) / total
            for modality, total in per_modality_total.items()
        },
        "confusion": confusion,
        "predictions": predictions,
    }


def format_agreement_markdown(result: Dict[str, object], baseline: float = 0.5) -> str:
    """Renders the agreement result as the markdown block Stage 7/8 report,
    including an explicit pass/fail against the sanity baseline."""
    rate = result["attribution_agreement_rate"]
    if rate != rate:  # NaN
        status = "UNKNOWN (no manipulated samples with a labeled modality)"
    elif rate > baseline:
        status = f"OK ({rate:.1%} > {baseline:.0%} baseline)"
    else:
        status = (
            f"BELOW BASELINE ({rate:.1%} <= {baseline:.0%}) -- explanations are not "
            "demonstrably better than chance at identifying the manipulated modality."
        )

    lines = [
        "## Attribution agreement",
        "",
        f"**Status:** {status}",
        "",
        f"Agreement rate: **{rate:.4f}** "
        f"({result['num_correct']}/{result['num_considered']} manipulated samples)",
        "",
        "| True modality | Agreement |",
        "|---|---|",
    ]
    for modality, value in sorted(result["per_modality_agreement"].items()):
        lines.append(f"| {modality} | {value:.4f} |")

    lines += [
        "",
        "### Confusion (true -> predicted)",
        "",
        "| True | Predicted | Count |",
        "|---|---|---|",
    ]
    for true_modality, predicted_counts in sorted(result["confusion"].items()):
        for predicted, count in sorted(predicted_counts.items()):
            lines.append(f"| {true_modality} | {predicted} | {count} |")

    return "\n".join(lines)
