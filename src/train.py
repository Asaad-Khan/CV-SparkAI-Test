import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102

from src.config import CFG
from src.utils.seed import set_seed
from src.utils.logger import log
from src.data.transforms import build_train_transforms, build_eval_transforms
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet18_tl import build_resnet18


def build_dataloaders(data_root: str, image_size: int, batch_size: int, num_workers: int):
    train_tf = build_train_transforms(image_size)
    eval_tf = build_eval_transforms(image_size)

    train_ds = Flowers102(root=data_root, split="train", download=True, transform=train_tf)
    val_ds = Flowers102(root=data_root, split="val", download=True, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += x.size(0)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * x.size(0)
        total += x.size(0)

    return running_loss / max(1, total)


def save_checkpoint(model: nn.Module, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / name
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def build_model(model_name: str, num_classes: int = 102) -> nn.Module:
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=True)
    raise ValueError(f"Unknown model: {model_name}")


def main(args: Dict):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    log(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, num_classes=102).to(device)

    # Optional: freeze backbone for transfer learning warmup
    if args.model == "resnet18" and args.freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        log("ResNet18 backbone frozen (fc head trainable).")

    # Optimize only trainable params at start
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    out_dir = Path(args.out_dir)
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze after warmup epochs
        if args.model == "resnet18" and args.freeze_backbone and epoch == (args.freeze_epochs + 1):
            log("Unfreezing ResNet18 backbone for fine-tuning.")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        log(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        # Save best checkpoint by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = save_checkpoint(model, out_dir, f"{args.model}_best.pt")
            log(f"Saved best checkpoint: {ckpt}")

    ckpt_final = save_checkpoint(model, out_dir, f"{args.model}_last.pt")
    log(f"Saved final checkpoint: {ckpt_final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default=CFG.data_root)
    parser.add_argument("--image_size", type=int, default=CFG.image_size)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)

    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "resnet18"])

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Transfer learning options (only used for resnet18)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_epochs", type=int, default=1)
    parser.add_argument("--lr_finetune", type=float, default=3e-4)

    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--out_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    main(args)
