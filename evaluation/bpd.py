# -*- coding: utf-8 -*-
"""
Compute Bits Per Dimension (BPD) as a log-likelihood proxy for generative diffusion models.

Author: Álvaro Duro
Date: 
"""


import torch
import torch.nn.functional as F
from typing import Callable

def compute_bpd(x_0: torch.Tensor,
                score_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                diffusion_process,
                eps: float = 1e-5) -> float:
    
    """
    Computes the Bits Per Dimension.

    This metric evaluates the generative model's ability to reconstruct original images 
    from the diffusion process.

    Args:
        x_0 (Tensor): Batch of original images with shape (batch_size, C, H, W), 
                    where pixel values are in the range [0, 1].
        score_model (nn.Module): Neural network that estimates the score ∇ log p(x_t | t).
        diffusion_process: Instance of a GaussianDiffusionProcess-like object with methods 
                        sigma_t(t), mu_t(x_0, t), and loss_function().
        eps (float): Small positive constant to avoid numerical instabilities at t=0.

    Returns:
        float: Average BPD (bits per dimension) over the batch.
    """

  
    batch_size = x_0.shape[0]
    t = torch.rand(batch_size, device=x_0.device) * (1.0 - eps) + eps
    z = torch.randn_like(x_0)

    sigma = diffusion_process.sigma_t(t).view(-1, 1, 1, 1).clamp(min=1e-6)  # Evita división por cero
    mu = diffusion_process.mu_t(x_0, t)
    x_t = mu + sigma * z

    score = score_model(x_t, t)
    loss_per_sample = ((score * sigma + z).pow(2) / sigma.pow(2)).sum(dim=(1, 2, 3))  # Corrección aquí
    loss = torch.mean(loss_per_sample)

    n_dim = x_0[0].numel()
    bpd = loss * 0.5 * (1.0 / torch.log(torch.tensor(2.0))) / n_dim
    return bpd.item()