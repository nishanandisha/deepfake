import math

from src.explain.attribution_agreement import (
    compute_attribution_agreement,
    format_agreement_markdown,
)
from src.explain.shap_modality import implicated_modality


def _split(visual_share: float) -> dict:
    return {
        "visual_share": visual_share,
        "acoustic_share": 1.0 - visual_share,
        "shap_visual": visual_share,
        "shap_acoustic": 1.0 - visual_share,
        "base_value": 0.0,
    }


def test_implicated_modality_thresholds():
    assert implicated_modality(_split(0.9)) == "video"
    assert implicated_modality(_split(0.1)) == "audio"
    assert implicated_modality(_split(0.5)) == "both"
    # Just inside the dominance margin -> still "both", not a modality call.
    assert implicated_modality(_split(0.6), dominance_margin=0.15) == "both"
    assert implicated_modality(_split(0.7), dominance_margin=0.15) == "video"


def test_perfect_agreement():
    splits = [_split(0.9), _split(0.1), _split(0.5)]
    truths = ["video", "audio", "both"]

    result = compute_attribution_agreement(splits, truths)

    assert result["attribution_agreement_rate"] == 1.0
    assert result["num_considered"] == 3
    assert result["num_correct"] == 3


def test_real_samples_are_excluded_from_agreement():
    splits = [_split(0.9), _split(0.9), _split(0.9)]
    truths = ["video", "none", "none"]  # only the first is a manipulated sample

    result = compute_attribution_agreement(splits, truths)

    assert result["num_considered"] == 1
    assert result["attribution_agreement_rate"] == 1.0
    # Predictions stay index-aligned with the input, with "" for skipped rows.
    assert result["predictions"] == ["video", "", ""]


def test_zero_agreement_when_modality_is_inverted():
    splits = [_split(0.05), _split(0.95)]
    truths = ["video", "audio"]  # model says the opposite modality each time

    result = compute_attribution_agreement(splits, truths)
    assert result["attribution_agreement_rate"] == 0.0


def test_per_modality_breakdown_and_confusion():
    splits = [_split(0.9), _split(0.9), _split(0.1)]
    truths = ["video", "audio", "audio"]

    result = compute_attribution_agreement(splits, truths)

    assert result["per_modality_agreement"]["video"] == 1.0
    assert result["per_modality_agreement"]["audio"] == 0.5
    assert result["confusion"]["audio"]["video"] == 1
    assert result["confusion"]["audio"]["audio"] == 1


def test_agreement_rate_is_nan_when_no_manipulated_samples():
    result = compute_attribution_agreement([_split(0.9)], ["none"])
    assert math.isnan(result["attribution_agreement_rate"])
    assert result["num_considered"] == 0


def test_markdown_flags_below_baseline():
    splits = [_split(0.05), _split(0.95)]
    truths = ["video", "audio"]
    result = compute_attribution_agreement(splits, truths)

    markdown = format_agreement_markdown(result, baseline=0.5)
    assert "BELOW BASELINE" in markdown


def test_markdown_reports_ok_above_baseline():
    splits = [_split(0.9), _split(0.1)]
    truths = ["video", "audio"]
    result = compute_attribution_agreement(splits, truths)

    markdown = format_agreement_markdown(result, baseline=0.5)
    assert "OK" in markdown
    assert "Attribution agreement" in markdown
