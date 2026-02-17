# Oxford Flowers 102 End-to-End Computer Vision Project

This repository implements a clean, end-to-end computer vision workflow (data inspection -> training -> evaluation -> inference) under realistic compute constraints.

---

## 1. Data Understanding & Preparation

### Dataset
**Oxford Flowers 102** (image classification, 102 classes), loaded via `torchvision.datasets.Flowers102` using the official `train/val/test` splits.

**Split sizes**
- Train: 1,020 images
- Val: 1,020 images
- Test: 6,149 images
- Total: 8,189 images

**Label format**
- Single-label classification (one integer class id per image)
- 102 unique classes detected

### Class distribution / imbalance
We computed per-class counts for each split and saved distributions:
- `reports/figures/class_distribution_train.png`
- `reports/figures/class_distribution_val.png`
- `reports/figures/class_distribution_test.png`
- CSV summary: `reports/class_counts_by_split.csv`

**Observed imbalance (from inspection script)**
- Train: min=10, max=10, max/min=1.00
- Val: min=10, max=10, max/min=1.00
- Test: min=20, max=238, max/min=11.90

**Important note about splits**
The provided `train` and `val` splits are perfectly balanced (10 images per class each), while the `test` split is imbalanced. This affects evaluation interpretation:
- Overall accuracy on test can be dominated by frequent classes.
- We report macro-averaged metrics (Macro-F1) and per-class statistics (via `classification_report`) to avoid misleading conclusions.

### Data quality checks
We performed an integrity scan by opening every image and counting failures:
- Corrupt/unreadable images (raw open): **0**

Qualitative sanity checks (saved grids):
- Random samples: `reports/figures/samples_random_raw.png`
- Minority-class samples: `reports/figures/samples_minority_raw.png`

### Image size characteristics
We measured raw image (W,H) sizes and saved:
- `reports/figures/image_size_distribution.png`

These statistics informed the preprocessing choice below.

### Preprocessing & augmentation plan
Inputs are standardized to `image_size=224`. We use ImageNet normalization because the stronger model fine-tunes an ImageNet-pretrained backbone.

- **Train transforms (augmentation):** mild `RandomResizedCrop`, horizontal flip, small rotation (±15°), mild color jitter
- **Eval transforms (val/test):** deterministic resize + center crop (no randomness)

Implementation: `src/data/transforms.py`

---

## 2. Model Selection, Training & Results

### Models
Two models were trained to provide a defensible baseline and a stronger transfer-learning approach:

1. **BaselineCNN (from scratch)**  
   A small CNN designed to train quickly on CPU/laptop GPU.  
   Code: `src/models/baseline_cnn.py`

2. **ResNet18 (transfer learning / fine-tuning)**  
   ResNet18 initialized with ImageNet weights, trained with a simple freeze→unfreeze strategy.  
   Code: `src/models/resnet18_tl.py`

### Training strategy (compute-aware)
- Training uses the official `train` split and selects the best checkpoint based on validation accuracy.
- ResNet18 uses a practical transfer learning warmup:
  - Freeze backbone initially (train only final layer)
  - Unfreeze and fine-tune with a smaller learning rate

Training entry point: `src/train.py`

### Evaluation (validation + test)
Evaluation script produces:
- Accuracy
- Macro-F1
- `classification_report` (per-class precision/recall/F1)
- Confusion matrix visualization
- Misclassified samples grid

Evaluation entry point: `src/evaluate.py`

**Results**

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|------:|-------------:|-------------:|--------------:|--------------:|
| BaselineCNN | 0.1529 | 0.0981 | 0.1257 | 0.0817 |
| ResNet18 TL | 0.7971 | 0.7811 | 0.7640 | 0.7602 |

Artifacts saved in `reports/`:
- Metrics JSON: `reports/metrics_<model>_<split>.json`
- Confusion matrix: `reports/confusion_matrix_<model>_<split>.png`
- Misclassified grid: `reports/misclassified_<model>_<split>.png`

---

---

## 6. FastAPI Deployment (Local)

A minimal FastAPI service is provided for production-oriented inference.

### Install API dependencies
```bash
python -m pip install -r requirements.txt


## 3. Project Structure

```text
cv-flowers102/
  src/
    data/
      dataset.py           # SafeFlowers102 wrapper (robust image loading)
      inspect_data.py      # Step-1 data inspection + plots + CSV
      transforms.py        # train/eval transforms
    models/
      baseline_cnn.py      # baseline model
      resnet18_tl.py       # transfer learning model builder
    utils/
      logger.py            # minimal timestamped logging
      seed.py              # reproducibility seed helper
    config.py              # paths + defaults (image size, batch size, etc.)
    train.py               # training pipeline (supports baseline/resnet18)
    evaluate.py            # metrics + confusion matrix + misclassified samples
    infer.py               # deployment-ready inference (top-k JSON)
  reports/
    figures/               # EDA figures from inspection step
  checkpoints/             # saved model checkpoints
  requirements.txt
  README.md
