# -*- coding: utf-8 -*-
"""
Variance Exploding (VE) diffusion process.

Author: Álvaro Duro y Carlos Beti
Date: 
"""

import torch
import numpy as np
from .gaussian_process import GaussianDiffusionProcess


class VEProcess(GaussianDiffusionProcess):
    """
    Variance Exploding (VE) diffusion process.

    Args:
        sigma: Controls the rate of variance explosion (default: 25.0)

    Example:
        >>> process = VEProcess(sigma=25.0)
        >>> x = torch.randn(2, 1, 4, 4)
        >>> t = torch.tensor([0.5, 0.5])
        >>> drift = process.drift_coefficient(x, t)
        >>> diffusion = process.diffusion_coefficient(t)
        >>> drift.shape
        torch.Size([2, 1, 4, 4])
        >>> diffusion.shape
        torch.Size([2])
    """
        
    def __init__(self, sigma: float = 25.0):
        drift_coefficient = lambda x_t, t: torch.zeros_like(x_t)
        diffusion_coefficient = lambda t: sigma ** t
        mu_t = lambda x_0, t: x_0
        sigma_t = lambda t: torch.sqrt(0.5 * (sigma ** (2 * t) - 1.0) / np.log(sigma))

        super().__init__(
            drift_coefficient=drift_coefficient,
            diffusion_coefficient=diffusion_coefficient,
            mu_t=mu_t,
            sigma_t=sigma_t,
        )


if __name__ == "__main__":
    import doctest
    doctest.testmod()
