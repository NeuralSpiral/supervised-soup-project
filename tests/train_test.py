
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
        lr=0.01
    )

    # Print summary of metrics
    print("History keys:", history.keys())
    print("Train Loss:", history["train_loss"])
    print("Val Accuracy:", history["val_acc"])
    print("F1 Score:", history["val_f1"])
    print("Top-5 Accuracy:", history["val_top5"])
    print("Confusion Matrix:\n", history["val_cm"][-1])

