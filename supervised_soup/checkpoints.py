"""
Module for checkpointing logic for training.

Provides functions to:
- save last checkpoint (local every epoch + wandb artifact every N epochs)
- best checkpoint (permanent W&B artifact)
- resume from latest checkpoint artifact
"""

import os
import torch
import wandb


def save_best_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    val_f1_macro: float,
    val_top5: float,
    val_roc_auc_macro: float,
    per_class_acc: dict,
):
    """
    Saves the best model checkpoint and uploads it as a permanent W&B artifact.
    Also updates W&B run summary with best-epoch metrics.
    """
    # path inside the W&B run directory
    checkpoint_path = os.path.join(
        wandb.run.dir,
        f"best_model_{wandb.run.name}.pt",
    )

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1_macro": val_f1_macro,
        "val_top5": val_top5,
        "val_roc_auc_macro": val_roc_auc_macro,
        "per_class_acc": per_class_acc,
    }

    torch.save(checkpoint, checkpoint_path)

    artifact = wandb.Artifact(
        name=f"best-model-{wandb.run.name}",
        type="model",
        description=f"Best model at epoch {epoch + 1}",
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)

    # update run summary with best metrics
    wandb.run.summary["best_epoch"] = epoch + 1
    wandb.run.summary["best_val_loss"] = val_loss
    wandb.run.summary["best_val_acc"] = val_acc
    wandb.run.summary["best_val_f1_macro"] = val_f1_macro
    wandb.run.summary["best_val_top5"] = val_top5
    wandb.run.summary["best_val_roc_auc_macro"] = val_roc_auc_macro

    for cls, acc in per_class_acc.items():
        wandb.run.summary[f"best/{cls}_acc"] = acc



def save_last_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: os.PathLike,
):
    """
    Saves the rolling last checkpoint locally.

    - Epoch must be the last completed epoch (0-indexed).
    - The file at `path` is overwritten every time.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def upload_last_checkpoint_artifact(
    *,
    local_checkpoint_path: os.PathLike,
    epoch: int,
    upload_interval: int,
):
    """
    Uploads the rolling last checkpoint as a W&B artifact every N epochs.

    Uses a fixed artifact name so `:latest` always resolves to the most
    recent uploaded checkpoint.
    """

    # only upload every N epochs, as defined in config.py 
    if (epoch + 1) % upload_interval != 0:
        return

    artifact = wandb.Artifact(
        name=f"last-checkpoint-{wandb.run.name}",
        type="checkpoint",
        description="Rolling last checkpoint",
    )
    artifact.add_file(str(local_checkpoint_path))
    wandb.log_artifact(artifact)




def try_resume_from_wandb(*, device: torch.device,):
    """
    Tries loading the latest checkpoint artifact from W&B.

    Returns:
        checkpoint dict if found, otherwise None.
    """
    try:
        artifact = wandb.use_artifact(
            f"last-checkpoint-{wandb.run.name}:latest",
            type="checkpoint",
        )
    except wandb.errors.CommError:
        # no artifact found or not reachable
        return None

    artifact_dir = artifact.download()
    checkpoint_path = os.path.join(artifact_dir, "last_checkpoint.pt")

    if not os.path.exists(checkpoint_path):
        return None

    return torch.load(checkpoint_path, map_location=device)


