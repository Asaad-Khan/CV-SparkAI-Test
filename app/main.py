from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import os

from src.config import CFG
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
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(model_name, num_classes=102)
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_topk(model: nn.Module, pil_img: Image.Image, device: torch.device, topk: int = 5, image_size: int = 224):
    tf = build_eval_transforms(image_size)
    x = tf(pil_img.convert("RGB")).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(topk, probs.numel())
    top_probs, top_idxs = torch.topk(probs, k=k)

    return [{"class_id": int(i), "prob": float(p)} for i, p in zip(top_idxs.tolist(), top_probs.tolist())]


class Prediction(BaseModel):
    class_id: int
    prob: float


class PredictResponse(BaseModel):
    model: str
    topk: List[Prediction]


app = FastAPI(title="Flowers102 Classifier API", version="1.0")

# Minimal config (can be moved to env vars later)
MODEL_NAME = os.getenv("MODEL_NAME", "resnet18")
CKPT_PATH = os.getenv("CKPT_PATH", "checkpoints/resnet18_best.pt")
DEFAULT_TOPK = int(os.getenv("TOPK", "5"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[nn.Module] = None


@app.on_event("startup")
def startup():
    global model
    try:
        model = load_model(MODEL_NAME, CKPT_PATH, device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model on startup: {e}")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "device": str(device), "model": MODEL_NAME}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), topk: Optional[int] = None):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    try:
        contents = await file.read()
        pil = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    k = int(topk) if topk is not None else DEFAULT_TOPK
    preds = predict_topk(model, pil, device, topk=k, image_size=CFG.image_size)

    return {"model": MODEL_NAME, "topk": preds}
