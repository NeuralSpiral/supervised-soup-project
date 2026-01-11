"""
LR scheduler functions.
"""

from __future__ import annotations

import torch.optim as optim


def build_scheduler(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    epochs: int,
    **kwargs,
):
    """
    LR schedulers.

    Supported schedulers:
      - none
      - cosine
      - cosine_warm
      - step
      - multistep
      - plateau
    """
    name = scheduler_name.lower()

    if name in ["none", "", "null"]:
        return None

    if name == "cosine":
        eta_min = kwargs.get("eta_min", 1e-6)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )

    if name == "cosine_warm":
        t_0 = kwargs.get("t_0", 10)
        t_mult = kwargs.get("t_mult", 1)
        eta_min = kwargs.get("eta_min", 1e-6)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
        )

    if name == "step":
        step_size = kwargs.get("step_size", max(1, epochs // 3))
        gamma = kwargs.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    if name == "multistep":
        milestones = kwargs.get(
            "milestones", [epochs // 2, int(epochs * 0.75)]
        )
        gamma = kwargs.get("gamma", 0.1)
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )

    if name == "plateau":
        factor = kwargs.get("factor", 0.1)
        patience = kwargs.get("patience", 3)
        min_lr = kwargs.get("min_lr", 1e-6)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    raise ValueError(
        f"Unknown scheduler: {scheduler_name}. "
        f"Choose from: none, cosine, cosine_warm, step, multistep, plateau."
    )
