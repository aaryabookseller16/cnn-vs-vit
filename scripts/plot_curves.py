import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"epoch", "val_acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def pretty_label(stem: str) -> str:
    """Convert checkpoint csv stem like `cnn_frac0.1_seed42_metrics` into a readable label."""
    m = re.match(r"^(cnn|vit)_frac([0-9.]+)_seed([0-9]+)_metrics$", stem)
    if not m:
        return stem

    model, frac, seed = m.group(1), float(m.group(2)), int(m.group(3))
    model_name = "CNN" if model == "cnn" else "ViT"

    # nice percent formatting
    pct = int(round(frac * 100))
    return f"{model_name} ({pct}% data, seed {seed})"


def main():
    ap = argparse.ArgumentParser(description="Plot validation accuracy learning curves.")
    ap.add_argument("--runs", nargs="+", required=True, help="Paths to CSV log files")
    ap.add_argument(
        "--out",
        default="artifacts/plots/learning_curves.png",
        help="Output image path (png/pdf/svg).",
    )
    ap.add_argument(
        "--title",
        default="CNN vs ViT on CIFAR-10 (learning curves)",
        help="Plot title",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="DPI for raster outputs (e.g., png)",
    )
    ap.add_argument(
        "--show_best",
        action="store_true",
        help="Mark each run's best validation accuracy point",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Slightly larger, presentation-friendly defaults
    plt.rcParams.update(
        {
            "figure.figsize": (9.5, 5.5),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
        }
    )

    fig, ax = plt.subplots()

    # Sort runs for stable legend ordering
    for p in sorted(args.runs):
        df = load_csv(p)
        label = pretty_label(Path(p).stem)

        ax.plot(df["epoch"], df["val_acc"], label=label)

        if args.show_best:
            best_idx = df["val_acc"].idxmax()
            best_epoch = int(df.loc[best_idx, "epoch"])
            best_acc = float(df.loc[best_idx, "val_acc"])
            ax.scatter([best_epoch], [best_acc], s=40)

    ax.set_title(args.title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

    # Put legend outside so it doesn't cover curves
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    fig.tight_layout()

    # Save main output
    save_kwargs = {"bbox_inches": "tight"}
    if out.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        save_kwargs["dpi"] = args.dpi

    fig.savefig(out, **save_kwargs)

    # Also save a vector PDF next to it for reports/slides, if output is raster
    if out.suffix.lower() == ".png":
        pdf_out = out.with_suffix(".pdf")
        fig.savefig(pdf_out, bbox_inches="tight")

    print(f"Saved: {out}")
    if out.suffix.lower() == ".png":
        print(f"Saved: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()