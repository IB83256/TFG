# -*- coding: utf-8 -*-
"""
Compute Inception Score using torchvision's official InceptionV3 API (modern version).

Author: Álvaro Duro
Date: 01-09-2026
"""

import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from typing import Tuple
import numpy as np

def preprocess_images(images: torch.Tensor, transform) -> torch.Tensor:

    """
    Resize and normalize images to fit InceptionV3 input.
    
    Args:
        images: Tensor (N, C, H, W) with values in [0, 1]
        transform: Transform pipeline from Inception_V3_Weights.DEFAULT.transforms()

    Returns:
        Preprocessed tensor
    """

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return transform(images)

def compute_inception_score(
    images: torch.Tensor,
    device: str = "cuda",
    splits: int = 10
) -> Tuple[float, float]:
    
    """
    Compute Inception Score (IS) for a batch of generated images.

    Args:
        images: Tensor of shape (N, C, H, W) in [0, 1]
        device: 'cuda' or 'cpu'
        splits: Number of splits for IS variance estimation

    Returns:
        Tuple (mean IS, std IS)
    """
    
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    model.to(device)
    model.eval()

    # Load and apply preprocessing transform
    transform = weights.transforms()
    images = preprocess_images(images, transform)
    images = images.to(device)

    with torch.no_grad():
        preds = []
        for i in range(0, images.size(0), 32):
            batch = images[i:i+32]
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    N = preds.shape[0]
    scores = []

    for i in range(splits):
        part = preds[i * N // splits: (i + 1) * N // splits, :]
        py = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-8) - np.log(py[None, :]))
        kl_sum = np.sum(kl, axis=1)
        scores.append(np.exp(np.mean(kl_sum)))

    return float(np.mean(scores)), float(np.std(scores))
