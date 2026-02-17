import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from PIL import Image

from src.config import CFG
from src.utils.logger import log
from src.data.transforms import build_eval_transforms
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet18_tl import build_resnet18


def build_model(model_name: str, num_classes: int = 102) -> nn.Module:
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "resnet18":
        # pretrained=False because we will load trained weights from checkpoint
        return build_resnet18(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, image_size: int) -> torch.Tensor:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(str(p)).convert("RGB")
    tf = build_eval_transforms(image_size)
    x = tf(img)  # CxHxW tensor
    return x.unsqueeze(0)  # 1xCxHxW


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: torch.device, topk: int = 5) -> Dict[str, Any]:
    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)  # [num_classes]

    k = min(topk, probs.numel())
    top_probs, top_idxs = torch.topk(probs, k=k)

    results = []
    for cls_idx, prob in zip(top_idxs.tolist(), top_probs.tolist()):
        results.append({"class_id": int(cls_idx), "prob": float(prob)})

    return {"topk": results}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    log(f"Device: {device}")

    model = build_model(args.model, num_classes=102)
    model = load_checkpoint(model, args.ckpt_path, device)

    x = preprocess_image(args.image_path, args.image_size)
    out = predict(model, x, device, topk=args.topk)

    # Add metadata useful for deployment/logging
    out["model"] = args.model
    out["checkpoint"] = args.ckpt_path
    out["image_path"] = args.image_path

    if args.pretty:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "resnet18"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=CFG.image_size)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--pretty", action="store_true")

    args = parser.parse_args()
    main(args)
