import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from src.config import CFG
from src.utils.seed import set_seed
from src.utils.logger import log
from src.data.dataset import SafeFlowers102


def ensure_dirs(figures_dir: Path, reports_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(counts: Counter, title: str, outpath: Path) -> None:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    plt.figure()
    plt.bar(range(len(labels)), values)
    plt.title(title)
    plt.xlabel("Class index")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_image_size_distribution(sizes: np.ndarray, outpath: Path) -> None:
    # sizes: Nx2 (W,H)
    w = sizes[:, 0]
    h = sizes[:, 1]

    plt.figure()
    plt.hist(w, bins=30, alpha=0.7, label="width")
    plt.hist(h, bins=30, alpha=0.7, label="height")
    plt.title("Image size distribution (raw)")
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_sample_grid(images, labels, outpath: Path, title: str, nrows: int = 5, ncols: int = 5) -> None:
    plt.figure(figsize=(10, 10))
    plt.suptitle(title)
    for i in range(nrows * ncols):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(images[i])
        ax.set_title(str(labels[i]), fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main(args) -> None:
    set_seed(args.seed)

    data_root = Path(args.data_root)
    reports_dir = Path(args.reports_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dirs(figures_dir, reports_dir)

    log(f"Data root: {data_root}")
    log("Loading Flowers102 splits (downloads if missing).")

    datasets = {
        "train": SafeFlowers102(root=str(data_root), split="train", download=True),
        "val": SafeFlowers102(root=str(data_root), split="val", download=True),
        "test": SafeFlowers102(root=str(data_root), split="test", download=True),
    }

    # ---- 1) Dataset size ----
    split_sizes = {k: len(v) for k, v in datasets.items()}
    total = sum(split_sizes.values())
    log(f"Split sizes: {split_sizes} | Total: {total}")

    # ---- 2) Label format + classes ----
    all_labels = []
    for split, ds in datasets.items():
        all_labels.extend(list(map(int, ds._labels)))

    unique_labels = sorted(set(all_labels))
    n_classes = len(unique_labels)
    log(f"Label format: integer class index (single-label classification).")
    log(f"Detected classes: {n_classes}")

    # ---- 3) Class distribution + imbalance stats ----
    split_counts = {}
    for split, ds in datasets.items():
        c = Counter(map(int, ds._labels))
        split_counts[split] = c

        plot_class_distribution(
            c,
            f"Class distribution ({split})",
            figures_dir / f"class_distribution_{split}.png",
        )

        vals = np.array(list(c.values()))
        min_cnt = int(vals.min())
        max_cnt = int(vals.max())
        ratio = float(max_cnt / max(1, min_cnt))

        top10 = c.most_common(10)
        bot10 = sorted(c.items(), key=lambda x: x[1])[:10]

        log(f"{split}: min={min_cnt}, max={max_cnt}, max/min={ratio:.2f}")
        log(f"{split}: top10 classes by count: {top10}")
        log(f"{split}: bottom10 classes by count: {bot10}")

    # ---- 4) Image size distribution (raw open) + corruption audit ----
    sizes = []
    corrupt = defaultdict(int)

    for split, ds in datasets.items():
        for i in tqdm(range(len(ds)), desc=f"Scanning {split}"):
            try:
                img_path = ds._image_files[i]
                pil = Image.open(img_path).convert("RGB")
                sizes.append(pil.size)  # (W,H)
            except Exception:
                corrupt[split] += 1

    sizes = np.array(sizes, dtype=np.int32)
    plot_image_size_distribution(sizes, figures_dir / "image_size_distribution.png")
    log(f"Corrupt/unreadable images (raw open): {dict(corrupt)}")

    # ---- 5) Qualitative audit grids (raw PIL) ----
    train_ds = datasets["train"]

    # Random samples
    idxs = np.random.choice(len(train_ds), size=25, replace=False)
    imgs, labs = [], []
    for idx in idxs:
        pil = Image.open(train_ds._image_files[idx]).convert("RGB")
        imgs.append(pil)
        labs.append(int(train_ds._labels[idx]))
    save_sample_grid(imgs, labs, figures_dir / "samples_random_raw.png", "Random samples (raw)")

    # Minority-class samples (from lowest-frequency train classes)
    train_counts = split_counts["train"]
    minority_classes = [k for k, _ in sorted(train_counts.items(), key=lambda x: x[1])[:10]]

    minority_idxs = []
    train_labels_arr = np.array(list(map(int, train_ds._labels)))

    for cls in minority_classes:
        cls_idxs = np.where(train_labels_arr == cls)[0]
        take = min(3, len(cls_idxs))
        minority_idxs.extend(list(np.random.choice(cls_idxs, size=take, replace=False)))

    minority_idxs = minority_idxs[:25]

    imgs, labs = [], []
    for idx in minority_idxs:
        pil = Image.open(train_ds._image_files[idx]).convert("RGB")
        imgs.append(pil)
        labs.append(int(train_ds._labels[idx]))

    # pad if not enough
    while len(imgs) < 25:
        idx = np.random.randint(0, len(train_ds))
        pil = Image.open(train_ds._image_files[idx]).convert("RGB")
        imgs.append(pil)
        labs.append(int(train_ds._labels[idx]))

    save_sample_grid(imgs, labs, figures_dir / "samples_minority_raw.png", "Minority-class samples (raw)")

    # ---- 6) Save class counts CSV (for README / analysis) ----
    rows = []
    for split, c in split_counts.items():
        for cls, cnt in c.items():
            rows.append({"split": split, "class": int(cls), "count": int(cnt)})

    df = pd.DataFrame(rows)
    out_csv = reports_dir / "class_counts_by_split.csv"
    df.to_csv(out_csv, index=False)
    log(f"Saved: {out_csv}")
    log(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=CFG.data_root)
    parser.add_argument("--reports_dir", type=str, default=CFG.reports_dir)
    parser.add_argument("--figures_dir", type=str, default=CFG.figures_dir)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    args = parser.parse_args()

    main(args)
