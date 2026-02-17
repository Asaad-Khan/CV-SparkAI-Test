import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.config import CFG
from src.utils.logger import log
from src.data.transforms import build_eval_transforms
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet18_tl import build_resnet18


def build_model(model_name: str, num_classes: int = 102) -> nn.Module:
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown model: {model_name}")


def load_model(model_name: str, ckpt_path: str, device: torch.device) -> nn.Module:
    model = build_model(model_name, num_classes=102)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: torch.device):
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_labels, all_preds


def plot_confusion_matrix(cm: np.ndarray, outpath: Path, title: str):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_misclassified_grid(dataset: Flowers102, y_true, y_pred, outpath: Path, max_items: int = 25):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        log("No misclassifications found. Skipping grid.")
        return

    idxs = wrong[:max_items]
    nrows, ncols = 5, 5

    plt.figure(figsize=(10, 10))
    plt.suptitle("Misclassified samples (true -> pred)")

    for i in range(nrows * ncols):
        ax = plt.subplot(nrows, ncols, i + 1)
        if i < len(idxs):
            idx = int(idxs[i])
            img_path = dataset._image_files[idx]
            pil = Image.open(img_path).convert("RGB")
            ax.imshow(pil)
            ax.set_title(f"{y_true[idx]}→{y_pred[idx]}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    log(f"Device: {device}")

    eval_tf = build_eval_transforms(args.image_size)
    ds = Flowers102(root=args.data_root, split=args.split, download=True, transform=eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_model(args.model, args.ckpt_path, device)

    y_true, y_pred = run_inference(model, loader, device)

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "model": args.model,
        "split": args.split,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"metrics_{args.model}_{args.split}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log(f"Saved metrics: {metrics_path}")
    log(f"{args.model} | {args.split} | Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(102)))
    plot_confusion_matrix(cm, out_dir / f"confusion_matrix_{args.model}_{args.split}.png",
                          f"Confusion Matrix ({args.model}, {args.split})")

    save_misclassified_grid(ds, y_true, y_pred, out_dir / f"misclassified_{args.model}_{args.split}.png")
    log(f"Saved confusion matrix + misclassified grid to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=CFG.data_root)
    parser.add_argument("--image_size", type=int, default=CFG.image_size)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)

    parser.add_argument("--model", type=str, required=True, choices=["baseline", "resnet18"])
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out_dir", type=str, default="reports")

    args = parser.parse_args()
    main(args)
