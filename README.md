# CNN vs Vision Transformer on Small Data (CIFAR-10)

## Motivation
Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) are two dominant paradigms for computer vision, but they behave very differently in **low-data regimes**.

This project was built to answer a simple but important question:

> *When data is scarce, why do CNNs often outperform Vision Transformers — and what does that look like in practice?*

Rather than relying on benchmark tables or pretrained models, this repository provides a **from-scratch, controlled comparison** between a custom CNN and a tiny Vision Transformer trained on CIFAR-10, with interpretability tools to visualize *how* each model reasons about images.

The goal is not just performance, but **understanding inductive bias, sample efficiency, and training dynamics** — the kinds of fundamentals that matter in real ML systems.

---

## What This Project Demonstrates

- ✅ Understanding of **inductive bias** (CNN locality vs Transformer global attention)
- ✅ Training neural networks **from scratch** in low-data and full-data regimes
- ✅ Clean PyTorch model design (modular CNN + minimal ViT implementation)
- ✅ Strong data augmentation and reproducible experiments
- ✅ Model interpretability via **Grad-CAM / attention visualizations**
- ✅ End-to-end engineering mindset (training → evaluation → deployment-ready structure)

This is designed to read as:
> *“I understand why these models behave differently — not just how to run them.”*

---

## Experiment Setup

### Dataset
- **CIFAR-10** (32×32 RGB images, 10 classes)
- Two regimes:
  - **Small-data**: 10% of the training set
  - **Full-data**: 100% of the training set

### Models

#### 1. Custom CNN (from scratch)
- Stacked Conv → BatchNorm → ReLU blocks
- MaxPooling + Global Average Pooling
- Dropout + Linear classifier
- Designed with strong **image-specific inductive bias**

#### 2. Tiny Vision Transformer (from scratch)
- Patch size = 4 (64 tokens per image)
- CLS token + learned positional embeddings
- Transformer encoder (multi-head self-attention + MLP)
- No pretraining (fair comparison)

Both models are trained using the **same data splits, augmentations, optimizer, and schedule**.

---

## Interpretability

To go beyond accuracy numbers, this project includes visual explanations:

- **Grad-CAM** for the CNN (last convolutional layer)
- **Attention-based visualizations** for the ViT (CLS-to-patch attention)

These help answer:
- What parts of the image does each model focus on?
- How does attention change under data scarcity?
- Where does the ViT fail without enough data?

---

## Repository Structure

```
small-data-cnn-vs-vit/
├── src/
│   ├── models/        # CNN and ViT implementations
│   ├── data/          # Dataset loading and splits
│   ├── train/         # Training loops and metrics
│   ├── interpret/     # Grad-CAM and attention maps
│   └── utils/         # Reproducibility and helpers
├── scripts/           # Train / eval / sanity-check scripts
├── artifacts/         # Logs, plots, checkpoints (gitignored)
├── app/               # FastAPI inference service
└── README.md
```

---

## Running the Project

### Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Sanity Checks
```bash
PYTHONPATH=. python scripts/cnn_check.py
PYTHONPATH=. python scripts/vit_check.py
```

### Training
```bash
PYTHONPATH=. python scripts/train.py --model cnn
PYTHONPATH=. python scripts/train.py --model vit
```

---

## Why This Matters

In many real-world settings (finance, healthcare, industrial ML), **data is limited** and **model choice matters**.

This project shows:
- why CNNs remain strong baselines
- when Transformers need scale or pretraining
- how architectural assumptions shape learning

It reflects how I think about ML systems: not as isolated models, but as **engineering + theory + evidence**.

---

## Author

**Aarya Bookseller**  
Computer Science @ Texas A&M University  
Interests: Machine Learning Systems, Representation Learning, and Applied AI

---

*This repository is intentionally kept simple, reproducible, and extensible — the way production ML code should be.*