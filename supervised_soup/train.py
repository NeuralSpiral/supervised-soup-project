"""
Implements train and eval loops.
Provides a run_training function.
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from supervised_soup.dataloader import get_dataloaders
import supervised_soup.config as config
from supervised_soup import seed as seed_module

from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix



# add NUM_CLASSES, EPOCHS, LR, OPTIMIZER to config?




# initializing model (Resnet-18)
# should be in model.py, not done in train
def build_model(num_classes=10, pretrained=True, freeze_layers=True):
    """Returns a ResNet-18 model with the last layer replaced for num_classes."""
    # not sure if V1 or V2 is better for baseline, or makes any difference
    weights = models.ResNet18_Weights.IMAGENET1K_V1  
    model = models.resnet18(weights=weights if pretrained else None)

    # to freeze or not to freeze
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    # replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(config.DEVICE)


# function that builds an optimizer
def build_optimizer(
    optimizer_name: str,
    params,
    lr: float,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
):
    """
    Builds optimizer based on optimizer_name.
    Not sure, which optimizers we want to use in the end, so I added the ones we've discussed.
    Supported:
        - "sgd"
        - "adam"
        - "adamw"
        - "adagrad"
        - "rmsprop"
    """
    name = optimizer_name.lower()

    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)

    elif name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    elif name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)

    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Choose from: sgd, adam, adamw, adagrad, rmsprop."
        )
    

# function that builds a scheduler
def build_scheduler(
    scheduler_name: str,
    optimizer,
    epochs: int,
    step_size: int = 10,
    gamma: float = 0.1,
):
    """
    Builds a learning rate scheduler.
    As far as I remember we've only discussed cosine annealing so far.
    Supported:
        - "none"
        - "cosine"
    """
    name = scheduler_name.lower()

    if name in ["none", "", "null"]:
        return None

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            f"Choose from: none, cosine."
        )
    



def save_checkpoint(model, optimizer, epoch, loss):
    """ Saves a model checkpoint"""

    # Create filename for checkpoints
    filename = f"checkpoint_epoch_{epoch:03d}.pt"
    path = config.CHECKPOINTS_PATH / filename
    
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


# training loop
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """ Trains the model for one epoch
    returns train_loss and train_acc for epoch"""

    model.train()

    running_loss = 0
    all_predictions = []
    all_labels = []

    # loop over batches in dataloader
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        # zeroes out previous gradients
        optimizer.zero_grad()
        # forward pass 
        outputs = model(imgs)
        # compute loss for batch
        loss = criterion(outputs, labels)
        # backprop gradients
        loss.backward()
        # update model parameters
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

        # get predicted labels and store predictions and labels
        predictions = outputs.argmax(dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # compute loss and accuracy for epoch
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_predictions)

    return epoch_loss, epoch_acc


# decorator to disable gradient calculation
@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    """ Validates the model for one epoch
    currrently returns: epoch_loss, epoch_acc, epoch_f1, epoch_top5, epoch_cm"""
    model.eval()
    running_loss = 0.0

    all_labels = []
    all_predictions = []
    # for top-k accuracy
    all_predicted_probabilities = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # forward pass
        # no gradient calculated for validation (decorator)
        outputs = model(images)

        # compute and accumulate loss
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        # get predicted classes
        predictions = outputs.argmax(dim=1)
        # store predictions, labels, and predicted probabilities
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_predicted_probabilities.extend(outputs.softmax(dim=1).cpu().numpy())

    # compute metrics for epoch
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    epoch_f1 = f1_score(all_labels, all_predictions, average="macro")
    epoch_top5 = float(top_k_accuracy_score(all_labels, all_predicted_probabilities, k=5))
    epoch_cm = confusion_matrix(all_labels, all_predictions)

    return epoch_loss, epoch_acc, epoch_f1, epoch_top5, epoch_cm


def run_training(
        epochs: int = 5, 
        with_augmentation: bool =False, 
        pretrained: bool = True,
        freeze_layers: bool = True,
        lr: float = 1e-3, 
        device: str = config.DEVICE,
        optimizer_name: str = "sgd",  # I put sgd as a default value
        scheduler_name: str = "cosine",
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
):
    """
    Main training function:
    - loads dataloaders
    - constructs model
    - loops over epochs
    - logs losses/accuracy
    - saves best checkpoint

    Example use:
        from supervised_soup.train import train_baseline
        results = train_baseline(epochs=10)
        optimizer_name="adamw",
        scheduler_name="cosine",
        lr=3e-4
    """
    # set seed for reproducibility
    seed_module.set_seed(config.SEED)

    # load Data
    train_loader, val_loader = get_dataloaders(
        with_augmentation=with_augmentation
    )

    model = build_model(num_classes=10, pretrained=pretrained, freeze_layers=freeze_layers)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    trainable_params = (p for p in model.parameters() if p.requires_grad)

    optimizer = build_optimizer(
        optimizer_name=optimizer_name,
        params=trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    scheduler = build_scheduler(
        scheduler_name=scheduler_name,
        optimizer=optimizer,
        epochs=epochs,
    )

    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_top5": [], "val_cm": [],
        "lr": [],
    }

    print(
        f"Starting training | optimizer={optimizer_name} | scheduler={scheduler_name} | "
        f"lr={lr} | epochs={epochs} | pretrained={pretrained} | freeze_layers={freeze_layers}"
    )

    for epoch in range(epochs):
        t0 = time.time()

        # loss and accuracy for training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # get loss and other metrics for validation
        val_loss, val_acc, val_f1, val_top5, val_cm = validate_one_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        # Logging placeholder for wb (or something else)
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"F1: {val_f1:.4f} | Top-5: {val_top5:.4f} "
            f"Time: {time.time() - t0:.1f}s"
        )

        # Save metrics, can use it later for plotting, visualizations, etc.
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_top5"].append(val_top5)
        history["val_cm"].append(val_cm)

        # Save best checkpoint
        # should we save best checkpoints for multiple/all metrics?
        # Maybe we should also save the last checkpoint always?
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss)

        if scheduler is not None:
            scheduler.step()

    print(f"Training complete. Best Validation Acc = {best_val_acc:.4f}")
    return model, history


