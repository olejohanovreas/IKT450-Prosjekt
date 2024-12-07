import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B7_Weights


def initialize_efficientnet(num_classes):
    # Load pre-trained EfficientNet model
    weights = EfficientNet_B7_Weights.DEFAULT
    model = models.efficientnet_b7(weights=weights)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model
