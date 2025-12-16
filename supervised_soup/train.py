"""
Implements train and eval loops.
Provides a run_training function.
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import wandb

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
    return model


def save_checkpoint(model, optimizer, epoch, val_loss):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    path = os.path.join(wandb.run.dir, "best_model.pt")
    torch.save(checkpoint, path)


# add per class acc
def per_class_accuracy(cm):
    """ Calculates per class accuracy from confusion matrix"""
    acc = {}
    for i in range(cm.shape[0]):
        correct = cm[i, i]
        total = cm[i].sum()
        acc[i] = correct / total if total > 0 else 0.0
    return acc


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
def validate_one_epoch(model, dataloader, criterion, device):
    """ Validates the model for one epoch
    currrently returns: epoch_loss, epoch_acc, epoch_f1, epoch_top5, epoch_cm"""
    model.eval()
    running_loss = 0.0

    all_labels = []
    all_predictions = []
    # for top-k accuracy
    all_predicted_probabilities = []

    for images, labels in dataloader:
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
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    epoch_f1 = f1_score(all_labels, all_predictions, average="macro")
    epoch_top5 = float(top_k_accuracy_score(all_labels, all_predicted_probabilities, k=5))
    epoch_cm = confusion_matrix(all_labels, all_predictions)

    return epoch_loss, epoch_acc, epoch_f1, epoch_top5, epoch_cm, all_labels, all_predictions


def run_training(epochs: int = 5, with_augmentation: bool =False, lr: float = 1e-3, device: str = config.DEVICE):
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
    """
    # set seed for reproducibility
    seed_module.set_seed(config.SEED)

    # load Data
    train_loader, val_loader = get_dataloaders(
        with_augmentation=with_augmentation
    )

    ## initialize wandb
    wandb.init(
        entity="supervised-soup",
        project="baseline-resnet-18",
        name=f"baseline_lr{lr}_aug{with_augmentation}",
        config={
            "model": "resnet18",
            "pretrained": True,
            "freeze_layers": True,
            "loss": "CrossEntropyLoss",
            "optimizer": "SGD",
            "scheduler": "CosineAnnealingLR",
            "learning_rate": lr,
            "min_lr": 1e-6,
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "augmentation": with_augmentation,
            "num_classes": 10,
            "seed": config.SEED,
        },
    )


    model = build_model(num_classes=10, pretrained=True, freeze_layers=True)
    model.to(device)

    wandb.watch(model, log="gradients", log_freq=100)

    # Loss function and optimizer: set to CrossEntropy and SGD for now
    criterion = nn.CrossEntropyLoss()
    # added filter to avoid iterating over frozen parameters
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_top5": [], "val_cm": []
    }

    # for montiroing when overfitting starts
    # overfitting defined as 5 consecutive epochs without validation improvement
    monitor = "val_loss" 
    patience = 5           
    best_val_metric = float("inf") 
    epochs_since_improvement = 0


    for epoch in range(epochs):
        t0 = time.time()

        # loss and accuracy for training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # get loss and other metrics for validation
        val_loss, val_acc, val_f1, val_top5, val_cm, val_labels, val_predictions = validate_one_epoch(model, val_loader, criterion, device)

        # update overfitting monitor
        current_metric = val_loss 

        if current_metric < best_val_metric:
            best_val_metric = current_metric
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1


        overfitting_flag = epochs_since_improvement >= patience
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        per_class_acc = per_class_accuracy(val_cm)



        ### wandb logging
        log_data = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/f1": val_f1,
            "val/top5": val_top5,
            "lr": current_lr,
            "epoch_time": time.time() - t0,
        }

        log_data.update({
            "diagnostics/best_val_loss": best_val_metric,
            "diagnostics/epochs_since_val_improvement": epochs_since_improvement,
            "diagnostics/overfitting_flag": int(overfitting_flag),
        })


        for cls, acc in per_class_acc.items():
            log_data[f"val/per_class_acc/class_{cls}"] = acc

        wandb.log(log_data)

        wandb.log({
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=val_labels,
                preds=val_predictions,
                class_names=[f"class_{i}" for i in range(10)],
            )
        })


        # Save best checkpoint (with wandb)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss)
            wandb.run.summary["best_val_acc"] = best_val_acc



        # prints to command line
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"F1: {val_f1:.4f} | Top-5: {val_top5:.4f} "
            f"LR: {current_lr:.6f} | "
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




    wandb.finish()
    print(f"Training complete. Best Validation Acc = {best_val_acc:.4f}")
    return model, history


