"""
Implements train and eval loops.
Provides a run_training function.
"""

import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import wandb

from supervised_soup.dataloader import get_dataloaders
import supervised_soup.config as config
from supervised_soup import seed as seed_module
from supervised_soup.models.model import build_model


from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize



# add EPOCHS, LR, OPTIMIZER to config?


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, val_f1_macro, val_top5, val_roc_auc_macro, per_class_acc):
    """
    - Saves the best model checkpoint and uploads it as a wandb artifact.
    - updates wandb summary with key metrics for the best epoch.
    """

    # path in run directory (temporary)
    checkpoint_path = os.path.join(wandb.run.dir, f"best_model_{wandb.run.name}.pt")

    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1_macro": val_f1_macro,
        "val_top5": val_top5,
        "roc_auc_macro": val_roc_auc_macro,
        "per_class_acc": per_class_acc,
    }
    torch.save(checkpoint, checkpoint_path)

    # Create wandb artifact (permanent)
    # We can use it for later analysis, or Grad-CAM
    artifact = wandb.Artifact(
        name=f"best_model_{wandb.run.name}",
        type="model",
        description=f"Best model at epoch {epoch}",
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)


    # log per-class accuracy to wandb summary
    for cls, acc in per_class_acc.items():
        wandb.run.summary[f"best/{cls}_acc"] = acc

    # log other metrics to wandb summary
    wandb.run.summary["best_val_loss"] = val_loss
    wandb.run.summary["best_val_acc"] = val_acc
    wandb.run.summary["best_val_f1_macro"] = val_f1_macro
    wandb.run.summary["best_val_top5"] = val_top5
    wandb.run.summary["best_val_roc_auc_macro"] = val_roc_auc_macro
    wandb.run.summary["best_epoch"] = epoch



# add per class acc
def per_class_accuracy(cm):
    """ Calculates per class accuracy from confusion matrix"""
    acc = {}
    for i in range(cm.shape[0]):
        correct = cm[i, i]
        total = cm[i].sum()
        acc[i] = correct / total if total > 0 else 0.0
    return acc


# log best cm (best epoch)
def log_best_confusion_matrix(
    *,
    y_true,
    y_pred,
    class_names,
    epoch,
):
    """
    Logs the best confusion matrix to W&B and saves it as an artifact.
    """
    # log visualization
    wandb.log({
        "best/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=class_names,
        )
    })

    # save raw cm
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(wandb.run.dir, "best_confusion_matrix.npy")
    np.save(cm_path, cm)

    # artifact
    cm_artifact = wandb.Artifact(
        name="best-confusion-matrix",
        type="evaluation",
        description=f"Validation confusion matrix at best epoch {epoch}",
    )
    cm_artifact.add_file(cm_path)
    wandb.log_artifact(cm_artifact)

    # metadata
    wandb.run.summary["best_epoch"] = epoch


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
    currrently returns: epoch_loss, epoch_acc, epoch_f1, epoch_top5, epoch_cm, roc_auc_macro_ovr, all_labels, all_predictions"""
    
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
    epoch_macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    epoch_top5 = float(top_k_accuracy_score(all_labels, all_predicted_probabilities, k=5))
    epoch_cm = confusion_matrix(all_labels, all_predictions)

    # ROC-AUC compute
    true_class_labels = np.array(all_labels)
    predicted_class_probabilities = np.array(all_predicted_probabilities)

    true_labels_one_hot = label_binarize(
        true_class_labels,
        classes=list(range(config.NUM_CLASSES))
    )

    # ROC-AUC macro (one-vs-rest)
    roc_auc_macro_ovr = roc_auc_score(
        true_labels_one_hot,
        predicted_class_probabilities,
        average="macro",
        multi_class="ovr",
    )

    return epoch_loss, epoch_acc, epoch_macro_f1, epoch_top5, epoch_cm, roc_auc_macro_ovr, all_labels, all_predictions


# TODO: refactor optimizer and scheduler creation to work with EXPERIMENT_CONFIG

# the * makes teh keyword arguments mandatory
def run_training(*, epochs: int = 5, with_augmentation: bool =False, pretrained: bool =True, freeze_layers: bool =True,  lr: float = 1e-3, device: str = config.DEVICE, seed: int = config.SEED,
                    # wandb (experiment metadata)
                    wandb_project: str = "x-AI-Proj-ImageClassification",
                    wandb_group: str | None = None,
                    wandb_name: str | None = None,
                    run_type: str = "baseline",
                     # for resuming from last checkpoint, in case
                     current_last_checkpoint_path: str | None = None ):
    """
    Main training function:
    - loads dataloaders
    - constructs model
    - loops over epochs
    - initializes wandb and tracks metrics
    - logs losses/accuracy, f1, cm, roc-auc
    - saves best checkpoint (currently based on val_loss)
        # change to macroF1?

    Example use:
        from supervised_soup.train import run_training
        
        run_training(epochs=10, lr=1e-3, wandb_group="baseline_frozen", wandb_name="seed42_lr1e-3_noaug")
    """
    # set seed for reproducibility
    seed_module.set_seed(seed)

    # load Data
    train_loader, val_loader = get_dataloaders(
        with_augmentation=with_augmentation
    )

    ## initialize wandb
    wandb.init(
        project=wandb_project,
        entity="neural-spi-university",
        group=wandb_group,
        name=wandb_name if wandb_name else f"{run_type}_lr{lr}_aug{with_augmentation}",
        config={
            "model": "resnet18",
            "pretrained": pretrained,
            "freeze_layers": freeze_layers,
            "loss": "CrossEntropyLoss",
            "optimizer": "SGD",
            "momentum": 0.9,
            "scheduler": "CosineAnnealingLR",
            "learning_rate": lr,
            "min_lr": 1e-6,
            "epochs": epochs,
            "batch_size": config.BATCH_SIZE,
            "augmentation": with_augmentation,
            "num_classes": config.NUM_CLASSES,
            "seed": seed,
        },
    )
    # wandb run-level metadata
    wandb.run.summary["run_type"] = run_type
    wandb.run.summary["model"] = "resnet18"
    wandb.run.summary["frozen_backbone"] = True


    model = build_model(num_classes=config.NUM_CLASSES, pretrained=pretrained, freeze_layers=freeze_layers)
    model.to(device)

    wandb.watch(model, log="gradients", log_freq=100)

    # Loss function and optimizer: set to CrossEntropy and SGD for now
    criterion = nn.CrossEntropyLoss()
    # added filter to avoid iterating over frozen parameters
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.9,)

    # for resuming from last checkpoint
    start_epoch = 0  # default
    if current_last_checkpoint_path is not None and os.path.exists(current_last_checkpoint_path):
        print(f"Resuming training from checkpoint: {current_last_checkpoint_path}")
        checkpoint = torch.load(current_last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6, last_epoch=start_epoch - 1)



    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1_macro": [], "val_top5": [], "val_cm": [], "val_roc_auc_macro": []
    }

    # for montiroing when overfitting starts
    # overfitting defined as 5 consecutive epochs without validation improvement
    patience = 5 
    best_val_loss = float("inf")       
    best_val_metric = best_val_loss
    epochs_since_improvement = 0

    best_epoch = None
    best_cm = None
    best_val_labels = None
    best_val_predictions = None



    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # loss and accuracy for training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # get loss and other metrics for validation
        val_loss, val_acc, val_f1_macro, val_top5, val_cm, val_roc_auc_macro, val_labels, val_predictions = validate_one_epoch(model, val_loader, criterion, device)

        # update overfitting
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
            "val/f1_macro": val_f1_macro,
            "val/top5": val_top5,
            "val/roc_auc_macro": val_roc_auc_macro,
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


        wandb.log(log_data, step=epoch)

        # save the last checkpoint (overwritten each epoch)
        current_last_checkpoint_path = os.path.join(wandb.run.dir, f"last_model_{wandb.run.name}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }, current_last_checkpoint_path)

        # Save best checkpoint and cm and store as artifact (with wandb)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1  

            # best confusion matrix
            best_cm = val_cm
            best_val_labels = val_labels
            best_val_predictions = val_predictions

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                val_acc,
                val_f1_macro,
                val_top5,
                val_roc_auc_macro,
                per_class_acc,  # pass from run_training
            )

            # log cm for best epoch
            log_best_confusion_matrix(
                y_true=best_val_labels,
                y_pred=best_val_predictions,
                class_names=[f"class_{i}" for i in range(10)],
                epoch=best_epoch,
            )

            wandb.run.summary["best_val_acc"] = best_val_acc
            wandb.run.summary["best_epoch"] = best_epoch          


        # prints to command line
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"F1: {val_f1_macro:.4f} | Top-5: {val_top5:.4f} | "
            f"ROC-AUC Macro: {val_roc_auc_macro:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {time.time() - t0:.1f}s"
        )

        # Save metrics, can use it later for plotting, visualizations, etc.
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1_macro)
        history["val_top5"].append(val_top5)
        history["val_cm"].append(val_cm)
        history["val_roc_auc_macro"].append(val_roc_auc_macro)




    wandb.finish()
    print(f"Training complete. Best Validation Acc was = {best_val_acc:.4f}.  Best Validation Loss was = {best_val_loss:.4f}")
    return model, history


