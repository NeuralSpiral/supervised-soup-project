"""
Provides a build_model function that returns a ResNet model with the last layer replaced for num_classes.
"""

import torch.nn as nn
from torchvision import models
import supervised_soup.config as config


RESNET_MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}

RESNET_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
}

FREEZE_STAGE_NAMES = {
    "conv1",
    "bn1",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
}


def _freeze_until(model, freeze_until: str):
    """
    Freeze all modules up to and including `freeze_until`.

    """
    if freeze_until not in FREEZE_STAGE_NAMES:
        raise ValueError(
            f"freeze_until must be one of {FREEZE_STAGE_NAMES}, "
            f"got {freeze_until}"
        )

    freezing = True
    for name, module in model.named_children():
        if freezing:
            for param in module.parameters():
                param.requires_grad = False
        if name == freeze_until:
            freezing = False


def print_trainable_layers(model):
    """
    Prints each top-level module name and whether it has trainable parameters.
    Useful to debug freeze_layers / freeze_until settings.
    """
    print("Trainable layers:")
    for name, module in model.named_children():
        trainable = any(p.requires_grad for p in module.parameters())
        print(f"  {name}: {'trainable' if trainable else 'frozen'}")


# Build the model
def build_model(
    model_name="resnet18",
    num_classes=config.NUM_CLASSES,
    pretrained=True,
    freeze_layers=True,
    freeze_until=None,
):
    """Returns a ResNet model with the last layer replaced for num_classes.
    - If pretrained = True, loads pretrained Imagenet weights (V1)
    - If freeze_layers = True, all layers will be frozen except the final layer
    - With freeze_until you can specify up to which layer to freeze (using the semantic resnet stage names)
    - freeze_until requires freeze_layers=False
    - model (not yet moved to device!)
    - How freeze_until works:
        - Freezes all ResNet stages up to and including the specified stage
        - Valid values: conv1, bn1, layer1, layer2, layer3, layer4
        - (unfreezing below layer1 or 2 probably won't make sense for us)
        - Example:
            freeze_until="layer3" -> only layer4 + fc trainable
            freeze_until="layer4" would be identical to freeze_layers=True
    """
    

    if model_name not in RESNET_MODELS:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Choose from {list(RESNET_MODELS.keys())}"
        )

    # initializes model with or without pretrained weights 
    if pretrained:
        # not sure if V1 or V2 is better, or makes any difference, should just stay consistent
        weights = RESNET_WEIGHTS[model_name]
    else:
        weights = None

    # select the model based on model_name
    model = RESNET_MODELS[model_name](weights=weights)

    # to freeze or not to freeze
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
    elif freeze_until is not None:
        _freeze_until(model, freeze_until)

    # get the number of input features
    in_features = model.fc.in_features
    print("fc input features:", in_features)

    # Replace the final layer with a new one for our dataset
    model.fc = nn.Linear(in_features, num_classes)

    # ensure classifier is always trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    print_trainable_layers(model)
    
    return model
