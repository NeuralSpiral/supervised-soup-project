"""
Tests for run_training() with different optimizers and schedulers.

Run from project root:
    python tests/train_test.py
"""


import time
from pathlib import Path
from supervised_soup.train import run_training
import supervised_soup.config as config

def test_run_training():
    """Quick test run for one epoch.
    if you run this on CPU it'll take like 20 minutes or so"""

    # Create a temporary folder for checkpoints
    test_checkpoint_path = Path("test_results")
    test_checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Temporarily override the CHECKPOINTS_PATH
    config.CHECKPOINTS_PATH = test_checkpoint_path
    model, history = run_training(
        epochs=1,            
        with_augmentation=False, 
        lr=0.01,
        optimizer_name="sgd",
        scheduler_name="cosine",
    )

    assert len(history["train_loss"]) == 1, "History not recorded correctly"
    assert len(history["val_acc"]) == 1, "Validation metrics missing"
    print("Default config test passed!")

def test_run_training_multiple_optimizers():
    """Test that run_training works with different optimizers."""
    optimizers = ["sgd", "adam", "adamw", "rmsprop", "adagrad"]

    for opt in optimizers:
        print(f"\n[TEST] Optimizer = {opt}")

        test_checkpoint_path = Path(f"test_results/checkpoints_{opt}")
        test_checkpoint_path.mkdir(parents=True, exist_ok=True)

        config.CHECKPOINTS_PATH = test_checkpoint_path

        model, history = run_training(
            epochs=1,
            with_augmentation=False,
            lr=0.001,
            optimizer_name=opt,
            scheduler_name="none",
        )

        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1
        print(f"Optimizer test passed: {opt}")


def test_run_training_scheduler_changes_lr():
    """Test that scheduler changes the learning rate."""
    print("\n[TEST] Scheduler = cosine (should change LR over epochs)")

    test_checkpoint_path = Path("test_results/checkpoints_scheduler_cosine")
    test_checkpoint_path.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINTS_PATH = test_checkpoint_path

    model, history = run_training(
        epochs=3,  # needs >1 epoch to see LR change
        with_augmentation=False,
        lr=0.01,
        optimizer_name="sgd",
        scheduler_name="cosine",
    )

    lr_values = history["lr"]
    print("LR values:", lr_values)

    # LR should change at least once
    assert len(set(lr_values)) > 1, "Learning rate did not change across epochs"
    print("Scheduler LR-change test passed!")


if __name__ == "__main__":
    history = test_run_training()
    test_run_training()
    test_run_training_multiple_optimizers()
    test_run_training_scheduler_changes_lr()

    print("All training tests passed!")

    # Print summary of metrics
    print("History keys:", history.keys())
    print("Train Loss:", history["train_loss"])
    print("Val Accuracy:", history["val_acc"])
    print("F1 Score:", history["val_f1"])
    print("Top-5 Accuracy:", history["val_top5"])
    print("Confusion Matrix:\n", history["val_cm"][-1])

