# -*- coding: utf-8 -*-
"""
Compute Fréchet Inception Distance (FID) using torchvision's official InceptionV3 API.

Author: Álvaro Duro
Date:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from scipy.linalg import sqrtm
import numpy as np
from typing import Tuple

def preprocess_images(images: torch.Tensor) -> torch.Tensor:

    """
    Prepares images for InceptionV3: resizes to 299x299, repeats channels if necessary, and rescales to [-1, 1].

    Args:
        images (Tensor): Input tensor of shape (N, C, H, W) with pixel values in [0, 1].

    Returns:
        Tensor: Output tensor of shape (N, 3, 299, 299), suitable for InceptionV3 input.
    """

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)  # Grayscale -> RGB

    resize = transforms.Resize((299, 299))
    images = resize(images)
    images = images * 2 - 1  # Escala a [-1, 1] como espera Inception
    return images

def get_activations(images: torch.Tensor, model: nn.Module, batch_size: int = 32) -> torch.Tensor:

    """
    Extracts feature activations from the penultimate layer of InceptionV3.

    Args:
        images (Tensor): Preprocessed image tensor of shape (N, 3, 299, 299).
        model (nn.Module): InceptionV3 model with final classification layer removed (i.e., model.fc = nn.Identity()).
        batch_size (int): Batch size used during inference.

    Returns:
        Tensor: Activation tensor of shape (N, 2048).
    """

    model.eval()
    activations = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(next(model.parameters()).device)
            pred = model(batch)
            activations.append(pred.cpu())
    return torch.cat(activations, dim=0)

def calculate_statistics(activations: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:

    """
    Computes empirical mean and covariance of a set of activations.

    Args:
        activations (Tensor): Tensor of shape (N, D).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean vector of shape (D,) and covariance matrix of shape (D, D).
    """

    mu = activations.mean(dim=0).numpy()
    sigma = torch.from_numpy(np.cov(activations.numpy(), rowvar=False))
    return mu, sigma.numpy()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2) -> float:

    """
    Computes the Fréchet distance (FID) between two multivariate Gaussian distributions.

    Args:
        mu1 (np.ndarray): Mean vector of the first distribution.
        sigma1 (np.ndarray): Covariance matrix of the first distribution.
        mu2 (np.ndarray): Mean vector of the second distribution.
        sigma2 (np.ndarray): Covariance matrix of the second distribution.

    Returns:
        float: Fréchet Inception Distance (FID) between the two distributions.
    """

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor, device: str = "cuda") -> float:
    """
    Calcula el Fréchet Inception Distance entre dos conjuntos de imágenes.
    Args:
        real_images: Tensor (N, C, H, W), valores en [0,1]
        generated_images: Tensor (N, C, H, W), valores en [0,1]
        device: 'cuda' o 'cpu'
    Returns:
        FID (float)
    """
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    model.fc = nn.Identity()
    model.to(device)

    real = preprocess_images(real_images)
    fake = preprocess_images(generated_images)

    act_real = get_activations(real, model)
    act_fake = get_activations(fake, model)

    mu_r, sigma_r = calculate_statistics(act_real)
    mu_g, sigma_g = calculate_statistics(act_fake)

    return calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

