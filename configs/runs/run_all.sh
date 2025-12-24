#!/usr/bin/env bash
set -euo pipefail

# Stream logs to terminal AND file, unbuffered so you see progress live
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

# Optional: pin these if your script supports them
SEED=42
BATCH=128

mkdir -p artifacts/stdout

run () {
  local name="$1"; shift
  echo "=============================="
  echo "RUN: ${name}"
  echo "CMD: $*"
  echo "=============================="
  "$@" 2>&1 | tee "artifacts/stdout/${name}.log"
}

# Run 1: CNN — 10% data
run "cnn-10pct" python scripts/train.py \
  --model cnn \
  --data_frac 0.1 \
  --epochs 20 \
  --batch_size ${BATCH} \
  --seed ${SEED} \
  --wandb \
  --run_name "cnn-10pct"

# Run 2: ViT — 10% data
run "vit-10pct" python scripts/train.py \
  --model vit \
  --data_frac 0.1 \
  --epochs 20 \
  --batch_size ${BATCH} \
  --seed ${SEED} \
  --wandb \
  --run_name "vit-10pct"

# Run 3: CNN — 100% data
run "cnn-100pct" python scripts/train.py \
  --model cnn \
  --data_frac 1.0 \
  --epochs 30 \
  --batch_size ${BATCH} \
  --seed ${SEED} \
  --wandb \
  --run_name "cnn-100pct"

# Run 4: ViT — 100% data
run "vit-100pct" python scripts/train.py \
  --model vit \
  --data_frac 1.0 \
  --epochs 30 \
  --batch_size ${BATCH} \
  --seed ${SEED} \
  --wandb \
  --run_name "vit-100pct"