"""Stage 7, part 3: Grad-CAM saliency over the visual branch's CNN backbone.

Per the build plan's architecture decision, the visual modality is
explained with CAM rather than SHAP -- SHAP over raw pixels is both
prohibitively expensive and produces attributions at the wrong granularity
for a moderator to act on. Grad-CAM instead highlights which spatial
regions of which frames drove the visual prediction.

Hooks the last conv feature map of the backbone (EfficientNet's
`features` block or ResNet's final residual stage), weights channels by
their gradient w.r.t. the visual logit, and ReLUs the result.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.models.common import masked_mean_pool


def _find_target_layer(visual_encoder) -> torch.nn.Module:
    """The last spatial (pre-pool) module of the backbone. VisualEncoder
    builds `backbone` as nn.Sequential(features, avgpool, flatten) for
    EfficientNet, or Sequential(*resnet_children, flatten) for ResNet -- in
    both cases index 0..-3 is the spatial trunk we want to hook."""
    backbone = visual_encoder.backbone
    spatial_trunk = backbone[0]
    # EfficientNet's `features` is itself a Sequential of blocks; hooking
    # its last block gives the final conv feature map.
    if isinstance(spatial_trunk, torch.nn.Sequential) and len(spatial_trunk) > 0:
        return spatial_trunk[-1]
    return spatial_trunk


class GradCAM:
    """Grad-CAM over a VisualEncoder's CNN backbone.

    Usage: cam = GradCAM(model.visual_encoder); heatmaps = cam(frames, mask)
    """

    def __init__(self, visual_encoder, target_layer: Optional[torch.nn.Module] = None):
        self.visual_encoder = visual_encoder
        self.target_layer = target_layer or _find_target_layer(visual_encoder)
        self._activations = None
        self._gradients = None
        self._handles = []

    def _register(self):
        def forward_hook(_module, _inputs, output):
            self._activations = output
            output.register_hook(self._save_gradient)

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))

    def _save_gradient(self, grad):
        self._gradients = grad

    def _remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(
        self,
        frames: torch.Tensor,
        padding_mask: torch.Tensor,
        head: torch.nn.Module,
    ) -> np.ndarray:
        """frames: [1, T, 3, H, W] (single sample). `head` maps the pooled
        encoder output to a scalar logit (the visual aux head).

        Returns per-frame heatmaps [T, H, W], each normalized to [0, 1].
        """
        assert frames.shape[0] == 1, "GradCAM expects one sample at a time"

        # The explanation pipeline runs on a *frozen* checkpoint (all
        # params requires_grad=False), and may be called inside an outer
        # no_grad block. Grad-CAM still needs a backward pass, so force
        # grad mode on and make the input require grad -- that alone makes
        # every downstream activation part of the autograd graph, without
        # unfreezing (and thus risking mutating) the model's parameters.
        self._register()
        try:
            with torch.enable_grad():
                frames = frames.detach().requires_grad_(True)

                self.visual_encoder.zero_grad(set_to_none=True)
                head.zero_grad(set_to_none=True)

                hv = self.visual_encoder(frames, padding_mask=padding_mask)
                logit = head(masked_mean_pool(hv, padding_mask)).squeeze()
                logit.backward()

            # activations/gradients: [T, C, h, w] (batch and time were
            # flattened together by VisualEncoder before the backbone).
            activations = self._activations.detach()
            gradients = self._gradients.detach()

            channel_weights = gradients.mean(dim=(2, 3), keepdim=True)  # GAP over space
            cam = torch.relu((channel_weights * activations).sum(dim=1))  # [T, h, w]
        finally:
            self._remove()

        frame_height, frame_width = frames.shape[-2:]
        heatmaps = []
        for frame_cam in cam.cpu().numpy():
            resized = cv2.resize(frame_cam, (frame_width, frame_height))
            span = resized.max() - resized.min()
            if span > 1e-12:
                normalized = (resized - resized.min()) / span
            else:
                normalized = np.zeros_like(resized)
            heatmaps.append(normalized)

        return np.stack(heatmaps)


def overlay_heatmap(
    frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Blends a [0,1] heatmap over an RGB uint8 frame as a JET colormap."""
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame_rgb, 1 - alpha, colored, alpha, 0)


def most_implicated_frames(heatmaps: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
    """Ranks frames by mean saliency; returns (frame_index, score) pairs."""
    scores = heatmaps.reshape(heatmaps.shape[0], -1).mean(axis=1)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [(int(idx), float(score)) for idx, score in ranked[:top_k]]
