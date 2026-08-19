"""Stage 7, part 2: fine-grained SHAP over the named acoustic features.

KernelSHAP over the ~47-dim per-frame acoustic descriptor vector from
Stage 3 (MFCCs + deltas, F0, voicing confidence, spectral stats, ZCR,
energy). Because the acoustic branch consumes a [S, D] sequence, features
are attributed at the descriptor level (a whole column across time), not
per-frame -- "F0 variance drove this decision", not "F0 at frame 137".

This is exactly why Stage 3 kept hand-crafted named features instead of a
learned embedding: every dimension here maps to a human-readable name from
`feature_names`.
"""

from typing import Dict, List

import numpy as np
import shap
import torch

from src.models.common import masked_mean_pool


def explain_acoustic_features(
    model,
    acoustic_features: torch.Tensor,
    acoustic_padding_mask: torch.Tensor,
    feature_names: List[str],
    background_features: torch.Tensor,
    device: torch.device,
    n_samples: int = 200,
) -> List[Dict[str, float]]:
    """Attributes the acoustic branch's own logit across named descriptors.

    acoustic_features: [B, S, D] for the samples being explained.
    background_features: [S, D] reference values (typically the per-column
    mean over a background set); a "missing" descriptor is replaced by its
    background column, the standard KernelSHAP absence treatment.

    Returns one {feature_name: shap_value} dict per sample.
    """
    model.eval()
    num_features = len(feature_names)
    assert acoustic_features.shape[-1] == num_features, (
        f"feature_names has {num_features} entries but tensor has "
        f"{acoustic_features.shape[-1]} columns -- ordering/count must match "
        "(see src/models/acoustic/features.py)"
    )

    def coalition_predict(masks: np.ndarray) -> np.ndarray:
        """masks: [n_coalitions, D] of 0/1 per descriptor column."""
        mask_tensor = torch.tensor(masks, dtype=torch.float32, device=device)
        n = mask_tensor.shape[0]

        # [n, S, D]: keep the sample's own column where present, else the
        # background column.
        sample_expanded = sample_features.unsqueeze(0).expand(n, -1, -1)
        background_expanded = background_features.unsqueeze(0).expand(n, -1, -1)
        column_mask = mask_tensor.unsqueeze(1)  # [n, 1, D] broadcasts over time
        mixed = torch.where(column_mask > 0.5, sample_expanded, background_expanded)

        mask_batch = sample_mask.unsqueeze(0).expand(n, -1)
        with torch.no_grad():
            ha = model.acoustic_encoder(mixed, padding_mask=mask_batch)
            logits = model.acoustic_aux_head(masked_mean_pool(ha, mask_batch)).squeeze(-1)
        return logits.cpu().numpy()

    background_features = background_features.to(device)
    results = []
    for i in range(acoustic_features.shape[0]):
        sample_features = acoustic_features[i].to(device)
        sample_mask = acoustic_padding_mask[i].to(device)

        explainer = shap.KernelExplainer(coalition_predict, np.zeros((1, num_features)))
        shap_values = explainer.shap_values(
            np.ones((1, num_features)), nsamples=n_samples, silent=True
        )
        shap_values = np.asarray(shap_values).reshape(-1)[:num_features]

        results.append(
            {name: float(value) for name, value in zip(feature_names, shap_values)}
        )

    return results


def top_k_features(
    attributions: Dict[str, float], k: int = 5
) -> List[tuple]:
    """Returns the k descriptors with the largest absolute SHAP value, as
    (name, value) pairs sorted by |value| descending."""
    return sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True)[:k]


def compute_background_features(background_loader, max_batches: int = 4) -> torch.Tensor:
    """Mean acoustic feature tensor [S, D] over background samples, used as
    the KernelSHAP reference."""
    total, count = None, 0
    for batch_idx, (_, _, features, _, _) in enumerate(background_loader):
        if batch_idx >= max_batches:
            break
        batch_sum = features.sum(0)
        total = batch_sum if total is None else total + batch_sum
        count += features.shape[0]

    if count == 0:
        raise ValueError("background_loader yielded no samples")
    return total / count
