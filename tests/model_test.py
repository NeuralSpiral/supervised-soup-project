"""
Updated tests for model.py

Run from project root with: 
    python -m tests.model_test


Tests:
- Multiple ResNet variants
- freeze_layers and freeze_until behavior
- Forward and backward passes
- Dataloader integration
"""


import torch
from models.model import build_model
from supervised_soup import config
from supervised_soup.dataloader import get_dataloaders


def test_forward_pass(model_name="resnet18", num_classes=10):
    """Tests forward pass with random inputs"""
    model = build_model(num_classes=num_classes, model_name=model_name, pretrained=False, freeze_layers=False)
    model.to(config.DEVICE)
    model.eval()

    x = torch.randn(2, 3, 224, 224, device=config.DEVICE)
    y = model(x)

    assert y.shape == (2, num_classes), f"{model_name} forward pass failed: got {y.shape}"
    print(f"{model_name} forward pass OK:", y.shape)


def test_backward_pass(num_classes=10):
    """Tests backward pass and ensures classifier has gradients"""
    model = build_model(num_classes=num_classes, pretrained=False, freeze_layers=False)
    model = model.to(config.DEVICE)
    model.train()

    x = torch.randn(4, 3, 224, 224, device=config.DEVICE)
    labels = torch.randint(0, num_classes, (4,), device=config.DEVICE)

    out = model(x)
    loss = torch.nn.CrossEntropyLoss()(out, labels)
    loss.backward()

    # Check that classifier is trainable
    assert model.fc.weight.grad is not None, "No gradient on model.fc.weight"
    print("Backward pass OK (classifier grads exist)")


def test_freeze_until(num_classes=10):
    """Tests that partial freezing with freeze_until works correctly"""
    freeze_until_stage = "layer2"
    model = build_model(num_classes=num_classes, freeze_layers=False, freeze_until=freeze_until_stage)
    model.to(config.DEVICE)

    print(f"\nTesting freeze_until={freeze_until_stage}")
    for name, module in model.named_children():
        trainable = any(p.requires_grad for p in module.parameters())
        print(f"{name}: {'trainable' if trainable else 'frozen'}")

    # fc should always be trainable
    assert all(p.requires_grad for p in model.fc.parameters()), "fc should always be trainable"


def test_freeze_layers(num_classes=10):
    """Tests that freeze_layers=True freezes all backbone layers"""
    model = build_model(num_classes=num_classes, freeze_layers=True)
    model.to(config.DEVICE)

    for name, p in model.named_parameters():
        if "fc" not in name:
            assert p.requires_grad is False, f"{name} should be frozen"
    print("freeze_layers=True test OK")


def test_dataloader(num_classes=10, batch_size=4):
    """Runs a batch from the dataloader through the model"""
    train_loader, _ = get_dataloaders(with_augmentation=False)
    model = build_model(num_classes=num_classes, pretrained=False)
    model.to(config.DEVICE)
    model.eval()

    images, labels = next(iter(train_loader))
    images = images.to(config.DEVICE)

    with torch.no_grad():
        outputs = model(images)

    assert outputs.shape == (images.shape[0], num_classes), f"Expected {(images.shape[0], num_classes)}, got {outputs.shape}"
    print("Dataloader forward pass OK:", outputs.shape)


if __name__ == "__main__":
    # Test multiple ResNet variants
    for model_name in ["resnet18", "resnet50"]:
        test_forward_pass(model_name=model_name)

    test_backward_pass()
    test_freeze_until()
    test_freeze_layers()
    test_dataloader()
