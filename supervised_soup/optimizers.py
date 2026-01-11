"""
Optimizer functions.
"""

from __future__ import annotations

import torch
import torch.optim as optim


def build_optimizer(
    optimizer_name: str,
    params,
    lr: float,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """
    Builds an optimizer based on optimizer_name.

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

    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)

    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(
        f"Unknown optimizer: {optimizer_name}. "
        f"Choose from: sgd, adam, adamw, adagrad, rmsprop."
    )
