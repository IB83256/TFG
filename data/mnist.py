from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
print("DATA_DIR:", DATA_DIR)

def get_mnist_loaders(
    batch_size,
    data_dir=DATA_DIR,
    normalize=False,
    num_workers=2,
):
    """
    Create MNIST training and test dataloaders.

    Args:
        batch_size:
            Training batch size.

        data_dir:
            Directory where MNIST is stored.

        normalize:
            If True, scale images from [0, 1] to [-1, 1].

        num_workers:
            Number of workers used by the dataloaders.

    Returns:
        Training and test dataloaders.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if normalize:
        transform = transforms.Compose([
        transforms.ToTensor(),                
        transforms.Normalize((0.5,), (0.5,)),  
    ])

    train_set = datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader