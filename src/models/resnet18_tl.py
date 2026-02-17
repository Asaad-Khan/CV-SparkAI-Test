import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 102, pretrained: bool = True) -> nn.Module:
    # New torchvision uses weights=... API; this keeps it compatible.
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
