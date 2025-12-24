# scripts/make_heatmaps.py
"""Generate and save heatmaps for CNN (Grad-CAM) and ViT (attention rollout)
using the SAME fixed CIFAR-10 test images.

Examples:
  PYTHONPATH=. python scripts/make_heatmaps.py \
    --model cnn \
    --ckpt artifacts/checkpoints/cnn_frac0.1_seed42_best.pt \
    --num_images 8 \
    --outdir artifacts/heatmaps

  PYTHONPATH=. python scripts/make_heatmaps.py \
    --model vit \
    --ckpt artifacts/checkpoints/vit_frac0.1_seed42_best.pt \
    --num_images 8 \
    --outdir artifacts/heatmaps
"""

import argparse
import importlib
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.data.cifar10 import make_loaders
from src.models.cnn import SmallCIFARCNN
from src.models.vit import TinyViT
from src.interpret.gradcam import GradCAM


# -----------------------------
# Small local helpers (no utils)
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] device = {device}", flush=True)
    return device


def overlay_heatmap(img_hwc: np.ndarray, heatmap_hw: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay heatmap on an image.
    img_hwc: float array (H,W,3) in [0,1]
    heatmap_hw: float array (H,W) (any scale; will be normalized)
    returns float array (H,W,3) in [0,1]
    """
    img = np.asarray(img_hwc, dtype=np.float32)
    hm = np.asarray(heatmap_hw, dtype=np.float32)

    # normalize heatmap -> [0,1]
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)

    # resize heatmap to match image if needed
    H, W = img.shape[:2]
    if hm.shape[0] != H or hm.shape[1] != W:
        hm_u8 = (hm * 255).astype(np.uint8)
        hm_u8 = np.array(Image.fromarray(hm_u8).resize((W, H), resample=Image.BILINEAR))
        hm = hm_u8.astype(np.float32) / 255.0

    # simple red-yellow style: red channel = hm, green = hm*0.6, blue = 0
    color = np.zeros_like(img)
    color[..., 0] = hm
    color[..., 1] = hm * 0.6
    color[..., 2] = 0.0

    out = (1.0 - alpha) * img + alpha * color
    return np.clip(out, 0.0, 1.0)


def try_make_test_loader():
    """Support older make_loaders signatures."""
    try:
        _tr, _va, te = make_loaders(batch_size=1, num_workers=0, data_frac=1.0, seed=42)
        return te
    except TypeError:
        _tr, _va, te = make_loaders(batch_size=1, num_workers=0)
        return te


def load_model(model_name: str, ckpt_path: Path, device: torch.device):
    if model_name == "cnn":
        model = SmallCIFARCNN(num_classes=10)
        # target the last conv-ish block in your CNN features stack
        target_layer = model.features[-3]
    elif model_name == "vit":
        model = TinyViT(num_classes=10)
        target_layer = None
    else:
        raise ValueError("model must be cnn or vit")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, target_layer


def vit_attention_heatmap(model, x: torch.Tensor) -> np.ndarray:
    """Return a ViT heatmap.

    If an explicit attention/rollout helper exists in `src/interpret/vit_attention.py`,
    we use it. Otherwise fall back to gradient-based saliency (works for any model).
    """
    # --- Try optional attention helper (if you implement one later) ---
    try:
        mod = importlib.import_module("src.interpret.vit_attention")
        candidates = [
            "vit_attention_map",
            "attention_rollout",
            "vit_rollout",
            "rollout",
            "attention_map",
            "get_attention_map",
        ]
        for name in candidates:
            if hasattr(mod, name) and callable(getattr(mod, name)):
                hm = getattr(mod, name)(model, x)
                if isinstance(hm, torch.Tensor):
                    hm = hm.detach().float().cpu().numpy()
                hm = np.asarray(hm)
                while hm.ndim > 2:
                    hm = hm[0]
                return hm
    except Exception:
        pass

    # --- Fallback: input-gradient saliency ---
    model.zero_grad(set_to_none=True)

    xg = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        logits = model(xg)
        pred = int(logits.argmax(dim=1).item())
        score = logits[:, pred].sum()
        score.backward()

        if xg.grad is None:
            raise RuntimeError("ViT saliency fallback failed: no gradients on input.")

        g = xg.grad.detach().abs().mean(dim=1)[0]  # (H, W)
        g = g - g.min()
        g = g / (g.max() + 1e-8)

    return g.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "vit"], required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--num_images", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="artifacts/heatmaps")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = pick_device()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    test_loader = try_make_test_loader()
    model, target_layer = load_model(args.model, Path(args.ckpt), device)

    cam = None
    if args.model == "cnn":
        cam = GradCAM(model=model, target_layer=target_layer)

    count = 0
    for x, _y in test_loader:
        if count >= args.num_images:
            break

        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())

        img = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        if args.model == "cnn":
            cam_map, _ = cam(x, class_idx=pred)   # cam_map: (B,H,W)
            heatmap = cam_map.squeeze(0).detach().cpu().numpy()
        else:
            heatmap = vit_attention_heatmap(model, x)

        overlay = overlay_heatmap(img, heatmap)

        Image.fromarray((overlay * 255).astype(np.uint8)).save(
            outdir / f"{args.model}_img{count}_pred{pred}.png"
        )
        count += 1

    print(f"Saved {count} heatmaps to {outdir}", flush=True)


if __name__ == "__main__":
    main()