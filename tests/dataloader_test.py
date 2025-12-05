
from supervised_soup.dataloader import get_dataloaders

import torch
import numpy as np
import matplotlib.pyplot as plt
import os



## Test batch loading to test that:
# Images have the shape ResNet-18 expects → [batch, 3, 224, 224]
# Labels are integer class IDs → [batch]
# Images are float32 → required by PyTorch models

def test_batch():

    train_loader, _ = get_dataloaders(with_augmentation=False)

    images, labels = next(iter(train_loader))

    print("Images:", images.shape)      
    print("Labels:", labels.shape)       
    print("Dtype:", images.dtype)
    print("Range:", images.min().item(), images.max().item())


# looping through an epoch to test the dataloader for training
def test_epoch_loading():
    train_loader, _ = get_dataloaders(with_augmentation=False)
    for i, (images, labels) in enumerate(train_loader):
        if i % 20 == 0:
            print(f"Batch {i} OK:", images.shape, labels.shape)


# print class names and mapping
def test_class_mapping():
    train_loader, val_loader = get_dataloaders(with_augmentation=False)
    dataset = train_loader.dataset
    print("Classes:", dataset.classes)
    print("Class → Index:", dataset.class_to_idx)


test_batch()
test_epoch_loading()
test_class_mapping()



##### visualizing some images 

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def unnormalize(img_tensor):
    """Undo normalization."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy() # CHW → HWC
    img = img * np.array(STD) + np.array(MEAN)  # undo normalization (std, mean)
    img = np.clip(img, 0, 1)
    return img


def show_image(img_tensor):
    """Show unnormalized image."""
    img = unnormalize(img_tensor)
    plt.imshow(img)
    plt.axis("off")


def visualize_batch():
    train_loader, _ = get_dataloaders(with_augmentation=False)
    images, labels = next(iter(train_loader))
    dataset = train_loader.dataset

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        show_image(images[i])

        # get filename from dataset.imgs
        # currently have to manually set shuffle=False in dataloader for correct filenames
        filepath = dataset.imgs[i][0]   
        filename = os.path.basename(filepath)

        plt.title(filename, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


visualize_batch()
