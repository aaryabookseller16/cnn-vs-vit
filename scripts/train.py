# scripts/train.py
import argparse
from pathlib import Path
import time
import csv

import torch

from src.train.loops import train_one_epoch, evaluate


def build_model(name: str):
    name = name.lower()
    if name == "cnn":
        from src.models.cnn import SmallCIFARCNN
        return SmallCIFARCNN(num_classes=10, dropout=0.0)
    elif name == "vit":
        from src.models.vit import TinyViT
        return TinyViT(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {name}. Use cnn|vit")


def try_make_loaders(*, batch_size: int, num_workers: int, data_frac: float, seed: int):
    from src.data.cifar10 import make_loaders

    try:
        return make_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            data_frac=data_frac,
            seed=seed,
        )
    except TypeError:
        return make_loaders(batch_size=batch_size, num_workers=num_workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "vit"], required=True)
    ap.add_argument("--data_frac", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="artifacts/checkpoints")

    # W&B
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="cnn-vs-vit")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)

    args = ap.parse_args()
    assert 0.0 < args.data_frac <= 1.0

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = try_make_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_frac=args.data_frac,
        seed=args.seed,
    )

    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_frac{args.data_frac}_seed{args.seed}"

    best_path = outdir / f"{tag}_best.pt"
    csv_path = outdir / f"{tag}_metrics.csv"

    # ---- CSV logging ----
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    # ---- W&B init (optional) ----
    use_wandb = False
    if args.wandb:
        try:
            import wandb

            use_wandb = True
            run_name = args.run_name or tag
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
            )
            wandb.watch(model, log="gradients", log_freq=200)
        except Exception as e:
            print(f"[WARN] W&B failed to initialize: {e}")
            use_wandb = False

    best_val = -1.0
    best_epoch = -1

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = evaluate(model, val_loader, device)

        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr.loss:.4f} acc {tr.acc:.4f} | "
            f"val loss {va.loss:.4f} acc {va.acc:.4f} | "
            f"{dt:.1f}s"
        )

        # CSV
        csv_writer.writerow([epoch, tr.loss, tr.acc, va.loss, va.acc])
        csv_file.flush()

        # W&B
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": tr.loss,
                    "train/acc": tr.acc,
                    "val/loss": va.loss,
                    "val/acc": va.acc,
                    "time/epoch_sec": dt,
                }
            )

        # Best checkpoint
        if va.acc > best_val:
            best_val = va.acc
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_val_acc": best_val,
                    "best_epoch": best_epoch,
                },
                best_path,
            )

    csv_file.close()

    # ---- Final test ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te = evaluate(model, test_loader, device)

    print(
        f"[BEST @ epoch {ckpt['best_epoch']}] "
        f"test loss {te.loss:.4f} acc {te.acc:.4f} | saved: {best_path}"
    )

    if use_wandb:
        wandb.summary["best/val_acc"] = float(ckpt["best_val_acc"])
        wandb.summary["best/epoch"] = int(ckpt["best_epoch"])
        wandb.log({"test/loss": te.loss, "test/acc": te.acc})
        wandb.finish()


if __name__ == "__main__":
    main()