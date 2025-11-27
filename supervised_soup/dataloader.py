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

import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np


# So, do I understand correctly that we import DATA_PATH from config, print it and 
# then overwrite it with an environment variable? Wouldn't it be better to not 
# don’t mix config and env var? 
# we can use config:
# from supervised_soup.config import DATA_PATH
# DATA_PATH = Path(DATA_PATH)
# or we can use the path from env:
# DATA_PATH = Path(os.environ["DATA_PATH"])


from supervised_soup.config import DATA_PATH

print(DATA_PATH)

# convert path string to Path
DATA_PATH = Path(os.environ["DATA_PATH"])



# transform images for baseline
baseline_transforms = transforms.Compose([
    transforms.Resize(256),       # resize shorter side to preserve ratio
    transforms.CenterCrop(224),   # shared size for ResNet
    transforms.ToTensor(),        # convert to float tensor [0,1]
    transforms.Normalize(         # standard ImageNet normalization (see docs)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# shall we add specific transformations for the training?
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = baseline_transforms


# loading the datasets for train and val with ImageFolder
# ImageFolder automatically reads and decodes JPEGs
train_dataset = datasets.ImageFolder(
    root=DATA_PATH / "train",
    transform=baseline_transforms
)

val_dataset = datasets.ImageFolder(
    root=DATA_PATH / "val",
    transform=baseline_transforms
)


# create batches
# not sure what batch size would be best for training on colab, we can adjust later
BATCH_SIZE = 64  

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # loads data in parallel
    num_workers=4,
    # should be true if using GPU, but false if CPU
    pin_memory=True 
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


## Test batch loading to test that:
# Images have the shape ResNet-18 expects → [batch, 3, 224, 224]
# Labels are integer class IDs → [batch]
# Images are float32 → required by PyTorch models

batch = next(iter(train_loader))
images, labels = batch

print("Batch loaded!")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Dtype:", images.dtype)
print("Min/Max:", images.min().item(), images.max().item())


# looping through an epoch to test the dataloader for training
for i, (images, labels) in enumerate(train_loader):
    if i % 20 == 0:
        print(f"Batch {i} OK:", images.shape, labels.shape)


# print class names and mapping
print("Classes:", train_dataset.classes)
print("Class → Index mapping:", train_dataset.class_to_idx)



# visualizing some images 

def show_image(img_tensor):
    """
    Visualizes images and checks if 
    - images are decoded correctly
    - colors are correct (RGB, not grayscale or BGR)
    - transforms didn’t break anything
    - normalization can be reversed
    """
    img = img_tensor.permute(1, 2, 0).numpy()      # CHW → HWC
    img = (img * 0.229 + 0.485)                    # undo normalization (std, mean)
    # Doesn't the line 151 undo only the red channel? Shouldn't we do the same for green and blue channels?
    # img = img * std + mean ?
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.axis("off")

# I think the code below can lead to the memory problems (as it's the top-level code and 
# every worker process will try to run it). Shall we put it inside a main guard? :
# if __name__ == "__main__":
#    images, labels = next(iter(train_loader))
#    ...


images, labels = next(iter(train_loader))
# to plot 9 images
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    show_image(images[i].cpu())
plt.show()




