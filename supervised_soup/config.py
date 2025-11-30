"""
Configures the data and results directories for our project, 
by loading environment variables from .env.

Use by importing with:
from supervised_soup.config import DATA_PATH, RESULTS_PATH

"""

# config for the DATA_PATH
import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")

if not DATA_PATH or not os.path.exists(DATA_PATH):
    print("Dataset path not found. Please check your .env file or Drive mount.")
else:
    print(f"Using dataset at: {DATA_PATH}")

# config for the RESULTS_PATH
RESULTS_PATH = os.getenv("RESULTS_PATH", "results")

# create results directory if it doesn't exist
os.makedirs(RESULTS_PATH, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(RESULTS_PATH)}")

# we might want to adjust these
BATCH_SIZE = 64
NUM_WORKERS = 4