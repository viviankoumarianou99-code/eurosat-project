import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    ResNet50 model for image classification.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B0 model for image classification.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_trainable_layers(model: nn.Module, train_mode: str = "head") -> None:
    """
    Works for both ResNet and EfficientNet:
    - head: only classifier head
    - last_block: last convolutional block + head
    - all: full fine-tuning
    """
    for p in model.parameters():
        p.requires_grad = False

    is_resnet = hasattr(model, "fc")
    is_effnet = hasattr(model, "classifier")

    def unfreeze_head():
        if is_resnet:
            for p in model.fc.parameters():
                p.requires_grad = True
        elif is_effnet:
            for p in model.classifier.parameters():
                p.requires_grad = True

    if train_mode == "head":
        unfreeze_head()

    elif train_mode == "last_block":
        if is_resnet:
            for p in model.layer4.parameters():
                p.requires_grad = True
            for p in model.fc.parameters():
                p.requires_grad = True
        elif is_effnet:
            for p in model.features[-1].parameters():
                p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True

    elif train_mode == "all":
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError("train_mode must be one of: head, last_block, all")


def get_device() -> torch.device:
    return torch.device("cpu")
