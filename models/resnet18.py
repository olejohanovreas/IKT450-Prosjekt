import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def initialize_resnet18(num_classes):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model
