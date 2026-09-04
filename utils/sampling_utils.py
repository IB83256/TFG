# -*- coding: utf-8 -*-
"""
Utility function to sample initial latent tensors for diffusion-based generation.

Author: Álvaro Duro
Date: 
"""

import torch
from typing import Tuple

def sample_initial_latents(
    n_images: int,
    image_shape: Tuple[int, int, int],  # (channels, height, width)
    mean: float = 0.0,
    std: float = 1.0,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Samples initial latent tensors x_T ~ N(mean, std^2 I)

    Args:
        n_images: Number of images to generate
        image_shape: Tuple (channels, height, width)
        mean: Mean of Gaussian noise
        std: Standard deviation of Gaussian noise
        device: Target device (CPU or CUDA)

    Returns:
        A tensor of shape (n_images, channels, height, width)
    """
    channels, height, width = image_shape
    return torch.randn(n_images, channels, height, width, device=device) * std + mean
