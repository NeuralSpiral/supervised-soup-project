#TODO: add/update docstrings

"""
This module implements the Dataloader by creating train/val 
DataLoader objects with the necessary preprocessing transforms for 
ResNet-18, including resizing, cropping, normalization, and batching.

The batch size is currently set to 64
-----

I have followed roughly these steps: 
1. Read the picture files.
2. Decode the JPEG content to RGB grids of pixels.
3. Convert these into floating-point tensors.
4. Resize them to a shared size.
5. Pack them into batches.

Source: https://deeplearningwithpython.io/chapters/chapter08_image-classification/

And the recommendations for resizing and normalizing for pre-trained models from the docs:
https://docs.pytorch.org/vision/0.12/models.html

"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# Needed for seed 
# import random
# import numpy as np

import supervised_soup.config as config


# Shall we also add a seed worker function for reproducibility?
# Each worker gets a different, but reproducible seed
# def seed_worker(worker_id: int):
#    worker_seed = torch.initial_seed() % 2**32
#    np.random.seed(worker_seed)
#    random.seed(worker_seed)



# Normalizations expected for pre-trained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# transform images for validation (always the same)
validation_transforms = transforms.Compose([
    transforms.Resize(256),       # resize shorter side to preserve ratio
    transforms.CenterCrop(224),   # shared size for ResNet
    transforms.ToTensor(),        # convert to float tensor [0,1]
    transforms.Normalize(         # standard ImageNet normalization (see docs)
        mean=MEAN,
        std=STD
    )
])

# transform images for baseline (no augmentations)
# since we are doing NO augmentations for the baseline, they are the same as validation transforms
baseline_transforms = validation_transforms

# transform images for later training (including augmentations)
# we can add and adjust the particular augmentations later
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])

def get_dataloaders(
    data_path=config.DATA_PATH,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    with_augmentation=False,
    # seed: int = 42,
):
    """
    Returns train_loader and val_loader.
    - Sets the transforms based on with_augmentation.
    - If with_augmentation=True we will use train_transforms (with augmentations)
    - If with_augmentation=False we use baseline transforms.
    - Validation transforms are always the same.

    - Example use for baseline:
    - train_loader, val_loader = get_dataloaders(with_augmentation=False)
    """

    data_path = Path(data_path)

    train_transform = train_transforms if with_augmentation else baseline_transforms
    # persistent = (num_workers > 0)  
    # so, I found this function that can speed the loading
    # the point is that normally, at the end of every epoch, the DataLoader kills the worker processes.
    # At the start of the next epoch, it creates them again. If we keep them alive it will be less costly for RAM
    # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110

    


    # loading the datasets for train and val with ImageFolder
    # ImageFolder automatically reads and decodes JPEGs
    train_dataset = datasets.ImageFolder(
        root=data_path / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=data_path / "val",
        transform=validation_transforms
    )

    # basically checks if GPU is available for training
    pin = torch.cuda.is_available()

    # seed
    # g = torch.Generator()
    # g.manual_seed(seed)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # loads data in parallel
        num_workers=num_workers,
        # should be true if using GPU, but false if CPU, PIN automatially sets it now depending whether CUDA is available
        pin_memory=pin,
        # persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader


