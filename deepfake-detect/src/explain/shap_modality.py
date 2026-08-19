"""Stage 7, part 1: coarse SHAP over the modality split.

Attributes the fused decision between the visual and acoustic pooled
vectors as two "features", via KernelSHAP. Deliberately NOT just reading
the gate weight: the gate is one input to the decision, but SHAP gives a
formal additive attribution of the actual model output, which is what the
"modality split" shown to moderators (and scored by the attribution
agreement rate) should be based on.

Background/reference values come from a set of background samples' pooled
vectors -- a "missing" modality is replaced by that background mean, the
standard KernelSHAP treatment of feature absence.
"""

from typing import Dict, List

import numpy as np
import shap
import torch


def _pooled_from_batch(model, batch, device: torch.device) -> tuple:
    frames, v_mask, features, a_mask, _ = batch
    with torch.no_grad():
        encoded = model.encode(
            frames.to(device), v_mask.to(device), features.to(device), a_mask.to(device)
        )
    return encoded["pooled_v"], encoded["pooled_a"]


def compute_background_pooled(model, background_loader, device: torch.device) -> tuple:
    """Mean pooled visual/acoustic vectors over background samples, used as
    the "feature absent" reference in KernelSHAP."""
    visual_sum, acoustic_sum, count = None, None, 0
    for batch in background_loader:
        pooled_v, pooled_a = _pooled_from_batch(model, batch, device)
        visual_sum = pooled_v.sum(0) if visual_sum is None else visual_sum + pooled_v.sum(0)
        acoustic_sum = pooled_a.sum(0) if acoustic_sum is None else acoustic_sum + pooled_a.sum(0)
        count += pooled_v.shape[0]

    if count == 0:
        raise ValueError("background_loader yielded no samples")
    return visual_sum / count, acoustic_sum / count


def explain_modality_split(
    model,
    pooled_v: torch.Tensor,
    pooled_a: torch.Tensor,
    background_pooled_v: torch.Tensor,
    background_pooled_a: torch.Tensor,
    device: torch.device,
    n_samples: int = 100,
) -> List[Dict[str, float]]:
    """KernelSHAP over the 2-feature coalition space {visual, acoustic}.

    Returns one dict per sample: shap_visual, shap_acoustic, base_value,
    and the normalized visual_share in [0,1] (the "modality split").
    """
    model.eval()

    def coalition_predict(masks: np.ndarray) -> np.ndarray:
        """masks: [n_coalitions, 2] of 0/1 for (visual present, acoustic
        present). Returns the fused logit for each coalition."""
        mask_tensor = torch.tensor(masks, dtype=torch.float32, device=device)
        n = mask_tensor.shape[0]

        visual_input = torch.where(
            mask_tensor[:, 0:1] > 0.5,
            sample_pooled_v.expand(n, -1),
            background_pooled_v.unsqueeze(0).expand(n, -1),
        )
        acoustic_input = torch.where(
            mask_tensor[:, 1:2] > 0.5,
            sample_pooled_a.expand(n, -1),
            background_pooled_a.unsqueeze(0).expand(n, -1),
        )

        with torch.no_grad():
            logits, _ = model.fuse_from_pooled(visual_input, acoustic_input)
        return logits.cpu().numpy()

    results = []
    for i in range(pooled_v.shape[0]):
        sample_pooled_v = pooled_v[i : i + 1].to(device)
        sample_pooled_a = pooled_a[i : i + 1].to(device)

        explainer = shap.KernelExplainer(coalition_predict, np.zeros((1, 2)))
        shap_values = explainer.shap_values(np.ones((1, 2)), nsamples=n_samples, silent=True)
        shap_values = np.asarray(shap_values).reshape(-1)[:2]

        shap_visual, shap_acoustic = float(shap_values[0]), float(shap_values[1])
        magnitude = abs(shap_visual) + abs(shap_acoustic)
        visual_share = abs(shap_visual) / magnitude if magnitude > 0 else 0.5

        results.append(
            {
                "shap_visual": shap_visual,
                "shap_acoustic": shap_acoustic,
                "base_value": float(np.asarray(explainer.expected_value).reshape(-1)[0]),
                "visual_share": visual_share,
                "acoustic_share": 1.0 - visual_share,
            }
        )

    return results


def implicated_modality(split: Dict[str, float], dominance_margin: float = 0.15) -> str:
    """Maps a modality split to a predicted manipulated_modality label
    comparable to the dataset's ground truth ("video", "audio", "both").

    If neither modality dominates by more than `dominance_margin` around an
    even 50/50 split, the sample is called "both" -- the model is saying
    the evidence came from both streams roughly equally.
    """
    visual_share = split["visual_share"]
    if visual_share > 0.5 + dominance_margin:
        return "video"
    if visual_share < 0.5 - dominance_margin:
        return "audio"
    return "both"
