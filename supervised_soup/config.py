"""
Configures the data and results directories for our project, 
by loading environment variables from .env.

Use by importing with:
from supervised_soup.config import DATA_PATH, RESULTS_PATH

"""

import os
import sys

from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()



def validate_path(path: Path) -> Path:
    """Validates the data path and raises a FileNotFoundError if not found."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}. Please check DATA_PATH in .env or Drive mount.")
    else:
        print(f"Using the following data path: {path}")
    return path


# get the data path from env
DATA_PATH = Path(os.getenv("DATA_PATH"))
# validate the path
DATA_PATH = validate_path(DATA_PATH)

# config for the RESULTS_PATH
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", "results"))
# create results directory if it doesn't exist
os.makedirs(RESULTS_PATH, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(RESULTS_PATH)}")

CHECKPOINTS_PATH = RESULTS_PATH / "checkpoints"
LOGS_PATH = RESULTS_PATH / "logs"
VISUALIZATIONS_PATH = RESULTS_PATH / "visualizations"
CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_PATH.mkdir(parents=True, exist_ok=True)



# we might want to adjust these
# we can set something like this and then override in .env. Just a suggestion::
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))     
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
# BATCH_SIZE = 64
# NUM_WORKERS = 4


# Check if GPU is available for training
CUDA = torch.cuda.is_available()  
print(f"Using CUDA: {CUDA}")
# set the device
DEVICE = torch.device("cuda" if CUDA else "cpu")

# test if running on colab
COLAB = 'google.colab' in sys.modules
if COLAB:
    print("Running on Colab. Make sure configurations are properly adjusted.")


# default seed for reproducibility
SEED = 42
