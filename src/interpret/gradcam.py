import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self.activations = out  # (B, C, H, W)

    def _save_grads(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # (B, C, H, W)

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor | None = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)  # (B, num_classes)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        scores = logits[torch.arange(logits.size(0)), class_idx]
        scores.sum().backward()

        # weights: (B, C)
        w = self.gradients.mean(dim=(2, 3))
        cam = (w[:, :, None, None] * self.activations).sum(dim=1)  # (B, H, W)
        cam = F.relu(cam)

        # normalize per-image
        cam_min = cam.flatten(1).min(dim=1)[0][:, None, None]
        cam_max = cam.flatten(1).max(dim=1)[0][:, None, None]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam, logits