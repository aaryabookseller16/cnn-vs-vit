

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class GradCAM:
    """Minimal Grad-CAM helper.

    Computes Grad-CAM for a given `target_layer`.

    Returns:
        cam: (B, H, W) float tensor in [0,1]
        logits: (B, num_classes)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Hooks to capture forward activations and backward gradients
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _module, _inp, out):
        # out: (B, C, H, W)
        self.activations = out

    def _save_grads(self, _module, _grad_input, grad_output):
        # grad_output[0]: (B, C, H, W)
        self.gradients = grad_output[0]

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[Union[int, torch.Tensor]] = None,
    ):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)  # (B, num_classes)

        # Resolve class indices for the batch
        if class_idx is None:
            class_idx_t = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx_t = torch.full(
                (logits.size(0),),
                class_idx,
                device=logits.device,
                dtype=torch.long,
            )
        else:
            class_idx_t = class_idx.to(device=logits.device)
            if class_idx_t.ndim == 0:
                class_idx_t = class_idx_t.view(1).repeat(logits.size(0))
            class_idx_t = class_idx_t.long()

        scores = logits[torch.arange(logits.size(0), device=logits.device), class_idx_t]
        scores.sum().backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "GradCAM did not capture gradients/activations. "
                "Check that target_layer is used in forward and supports backward hooks."
            )

        # Channel-wise weights: global-average-pool gradients
        # w: (B, C)
        w = self.gradients.mean(dim=(2, 3))

        # Weighted sum of activations
        # cam: (B, H, W)
        cam = (w[:, :, None, None] * self.activations).sum(dim=1)
        cam = F.relu(cam)

        # Normalize per-image into [0,1]
        cam_min = cam.flatten(1).min(dim=1)[0][:, None, None]
        cam_max = cam.flatten(1).max(dim=1)[0][:, None, None]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, logits