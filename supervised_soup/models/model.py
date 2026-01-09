"""
Provides a build_model function that returns a ResNet-18 model with the last layer replaced for num_classes.
    
"""


import torch.nn as nn
from torchvision import models
import supervised_soup.config as config

# Build the model
def build_model(num_classes=config.NUM_CLASSES, pretrained=True, freeze_layers=True):
    """Returns a ResNet-18 model with the last layer replaced for num_classes.
    - If pretrained = True, loads pretrained Imagenet weights (V1)
    - If freeze layers = True, all layers will be frozen except the final layer
    - model (not yet moved to device!)"""
    
    # initializes model with or without pretrained weights 
    if pretrained:
        # not sure if V1 or V2 is better , or makes any difference, should just stay consistent
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.resnet18(weights=weights)

    # to freeze or not to freeze
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    # get the number of input features
    in_features = model.fc.in_features
    print("fc input features:", in_features)

    # Replace the final layer with a new one for our dataset
    model.fc = nn.Linear(in_features, num_classes)
    return model

