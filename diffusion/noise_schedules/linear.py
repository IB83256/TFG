# -*- coding: utf-8 -*-
"""
Linear beta schedule for variance preserving diffusion models (OOP version).

Author: Álvaro Duro 
Date: 
"""

from .base_schedule import NoiseSchedule
from torch import Tensor
import torch

class LinearSchedule(NoiseSchedule):

    """
    Linear beta schedule class for variance preserving diffusion models with general time interval [0, T].

    Args:
        beta_min: Minimum beta value.
        beta_max: Maximum beta value.
        T: Final time of the diffusion process (default: 1.0).

    Methods:
        alphas_cumprod(t): Computes cumulative alpha from t.
        integrated_beta(t): Computes the integral of beta from 0 to t.
        beta(t): Computes beta at time t.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def alphas_cumprod(self, t: Tensor) -> Tensor:

        """
        Computes the cumulative product of alpha values for the given time tensor t.

        Implements:
            ᾱ(t) = exp(-∫₀ᵗ β(s) ds)

        Args:
            t (Tensor): Tensor of time values in [0, T].

        Returns:
            Tensor: Cumulative product of alphas evaluated at t.

        Example:
            >>> schedule = LinearSchedule(beta_min=0.1, beta_max=0.5, T=1.0)
            >>> t = torch.tensor([0.0, 0.5, 1.0])
            >>> schedule.alphas_cumprod(t)
            tensor([1.0000, 0.9048, 0.7408])
        """

        return torch.exp(-self.integrated_beta(t))

    def integrated_beta(self, t: Tensor) -> Tensor:

        """
        Computes the integral of the linear beta schedule from 0 to t.

        Implements:
            ∫₀ᵗ β(s) ds = β_min * t + 0.5 * (β_max - β_min) * t² / T

        Args:
            t (Tensor): Tensor of time values in [0, T].

        Returns:
            Tensor: Integrated beta values evaluated at each time t.
        """
            
        t_norm = t / self.T
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t_norm * t

    def beta(self, t: Tensor) -> Tensor:

        """
        Computes the instantaneous beta(t) for the linear schedule.

        Implements:
            β(t) = β_min + (β_max - β_min) * (t / T)

        Args:
            t (Tensor): Tensor of time values in [0, T].

        Returns:
            Tensor: Instantaneous beta values.

        Example:
            >>> schedule = LinearSchedule(beta_min=0.1, beta_max=0.5, T=1.0)
            >>> t = torch.tensor([0.0, 0.5, 1.0])
            >>> schedule.beta(t)
            tensor([0.1000, 0.3000, 0.5000])
        """

        t_norm = t / self.T
        return self.beta_min + (self.beta_max - self.beta_min) * t_norm

if __name__ == "__main__":
    import doctest
    doctest.testmod()
