#TODO: add/update docstrings

"""
This module implements the Dataloader by creating train/val 
DataLoader objects with the necessary preprocessing transforms for 
ResNet-18, including resizing, cropping, normalization, and batching.

The batch size is currently set to 64

Augmentation presets:
- random crop + horizontal flip + color/lighting changes
- rotation / translation
- noise injection
- random erasing / cutout-like occlusion
- optional AutoAugment(ImageNet)
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
from torchvision.transforms import InterpolationMode

import supervised_soup.config as config
import supervised_soup.seed as seed_module


# global seed
seed_module.set_seed(config.SEED)

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

<<<<<<< HEAD
# transform images for later training (including augmentations)
# we can add and adjust the particular augmentations later
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])
=======
# Helpers for augmentations
def _gaussian_noise(std: float = 0.02):
    """
    Adds gaussian noise to a tensor image.
    Assumes input is a float tensor in [0, 1] before normalization.
    """
    def _fn(x: torch.Tensor) -> torch.Tensor:
        if std <= 0:
            return x
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0.0, 1.0)
    return _fn


def build_train_transforms(
    preset: str = "light",
    *,
    noise_std: float = 0.02,
    random_erasing_p: float = 0.25,
):
    """
    Build train transforms based on a preset.

    Presets:
      - "light": crop + flip + mild color jitter
      - "strong": light + rotation + translation + noise + random erasing
      - "autoaugment": crop + flip + AutoAugment(ImageNet) + random erasing
    """
    preset = (preset or "light").lower()

    if preset not in {"light", "strong", "autoaugment"}:
        raise ValueError(
            f"Unknown augmentation preset: {preset}. "
            f"Choose from: light, strong, autoaugment."
        )

    common = [
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
    ]

    if preset in {"light", "strong"}:
        # stand-in for “color jittering / lighting changes”
        color = [
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            )
        ]
    else:
        color = []

    if preset == "strong":
        geom = [
            # rotation 
            transforms.RandomRotation(degrees=25, interpolation=InterpolationMode.BILINEAR),
            # translation / shifting
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
        ]
        noise = [transforms.Lambda(_gaussian_noise(std=noise_std))]
        auto = []
    elif preset == "autoaugment":
        geom = []
        noise = []
        auto = [transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)]
    else:
        geom = []
        noise = []
        auto = []

    tail = [
        transforms.ToTensor(),
        *noise,
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(
            p=random_erasing_p,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ]

    return transforms.Compose(common + color + geom + auto + tail)

# default augmented transforms 
train_transforms = build_train_transforms("light")
>>>>>>> 099a4ca (Added new augmentations)

def get_dataloaders(
    data_path=config.DATA_PATH,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    with_augmentation=False,
    seed=config.SEED,
    augmentation_preset: str = "light",
    augmentation_kwargs: dict | None = None,
):
    """
    Returns train_loader and val_loader.
    - Sets the transforms based on with_augmentation.
    - If with_augmentation=True -> build_train_transforms(augmentation_preset, **augmentation_kwargs).
    - If with_augmentation=False -> baseline_transforms.
    - Validation transforms are always the same.
    - Configured for reproducibility when randomness is involved, e.g. shuffling and augmentations.
    - Uses functionality from seed.py for reproducibility.

    - Example use for baseline:
    - train_loader, val_loader = get_dataloaders(with_augmentation=False)

    New:
    - augmentation_preset: "light" | "strong" | "autoaugment"
    - augmentation_kwargs: passed to build_train_transforms (e.g., noise_std, random_erasing_p)

    """

    data_path = Path(data_path)

    if with_augmentation:
        augmentation_kwargs = augmentation_kwargs or {}
        train_transform = build_train_transforms(augmentation_preset, **augmentation_kwargs)
    else:
        train_transform = baseline_transforms
    
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

    # defines a generator for deterministic shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # loads data in parallel
        num_workers=num_workers,
        # should be true if using GPU, but false if CPU, PIN automatially sets it now depending whether CUDA is available
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_module.seed_worker,
        generator=generator,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    return train_loader, val_loader


