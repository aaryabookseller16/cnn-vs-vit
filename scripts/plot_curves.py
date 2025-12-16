import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_csv(path):
    df = pd.read_csv(path)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="paths to CSV log files")
    ap.add_argument("--out", default="artifacts/plots/learning_curves.png")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for p in args.runs:
        df = load_csv(p)
        label = Path(p).stem
        plt.plot(df["epoch"], df["val_acc"], label=f"{label} (val acc)")

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved:", out)

if __name__ == "__main__":
    main()