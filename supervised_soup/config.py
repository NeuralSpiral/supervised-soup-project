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

# Do we want path as a String? We can also use pathlib.Path?
# DATA_PATH = os.getenv("DATA_PATH")
# if not DATA_PATH:
#    raise RuntimeError("DATA_PATH is not set in .env")
#
# DATA_PATH = Path(DATA_PATH)


DATA_PATH = os.getenv("DATA_PATH")

# As far as I understand the 1st print below, we only get a warning if the path isn't found.
# But the program continues to run. Isn't it better to raise an error here and fail faster?
# Something like? : 
# if not DATA_PATH or not os.path.exists(DATA_PATH):
#   raise RuntimeError("Dataset path isn't set or doen't exist. Check DATA_PATH in .env")
# else:
#   print(f"Using dataset at: {DATA_PATH}")


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

# we can set something like this and then override in .env. Just a suggestion::
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))     
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

BATCH_SIZE = 64
NUM_WORKERS = 4