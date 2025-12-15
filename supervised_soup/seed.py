"""
Module for reproducibility. Setting seeds and other related issues for avoiding randomness.
"""
# https://docs.pytorch.org/docs/stable/notes/randomness.html
# we may want to add more here after checking these checklists: https://github.com/paperswithcode/releasing-research-code
# and the other things I linked on confluence for reproducibility and ML research


import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Sets a global seed for reproducibility in random, numpy, and torch.
    Adjusts Pytorch settings to avoid randomness.
    Should be called at the start of training scripts.
    """

    # set global seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN settings for reproducibility, may slow down training slightly
    # see: https://docs.pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # this might be stricter then we want/need, might break RandomResizedCrop
    torch.use_deterministic_algorithms(True)


# For DataLoader(worker_init_fn=seed_worker)
def seed_worker(worker_id):
    """ Make seed workers for DataLoader reproducible.
    Assigns explicit seed to each worker.
    use in DataLoader(worker_init_fn=seed_worker)"""

    # https://docs.pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed
    # NumPy expects 32-bit seed
    np.random.seed(seed % 2**32)  
    random.seed(seed)
