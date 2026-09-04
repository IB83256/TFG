# -*- coding: utf-8 -*-
"""
Data loader for MNIST and CIFAR-10 with optional class filtering and custom transforms.

Author: Álvaro Duro 
Date: 
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Optional, Callable, Union

def load_dataset(
    name: str,
    batch_size: int = 128,
    class_label: Optional[int] = None,
    transform: Optional[Callable] = None,
    return_loader: bool = True,
    num_threads: int = 1
) -> Union[DataLoader, torch.utils.data.Dataset]:
    """
    Loads MNIST or CIFAR-10 training set with optional class filtering and transform.

    Args:
        name (str): 'mnist' or 'cifar10'
        batch_size (int): Batch size for DataLoader
        class_label (int or None): Filter the dataset to include only this class (0–9)
        transform (Callable or None): Custom transform to apply to the images
        return_loader (bool): If True, returns DataLoader; otherwise returns raw Dataset
        num_threads (int): Number of subprocesses used by DataLoader

    Returns:
        DataLoader or Dataset depending on return_loader
    """
    name = name.lower()
    if transform is None:
        transform = ToTensor()

    if name == "mnist":
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        targets = dataset.targets

    elif name == "cifar10":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        targets = torch.tensor(dataset.targets)
        
    else:
        raise ValueError("Only 'mnist' and 'cifar10' are supported.")

    if class_label is not None:
        indices = torch.where(targets == class_label)[0]
        dataset = Subset(dataset, indices)

    if return_loader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads)
    
    else:
        return dataset
