# -*- coding: utf-8 -*-
"""
Cosine beta schedule for variance preserving diffusion models (OOP version).

Author: Álvaro Duro 
Date: 
"""

from .base_schedule import NoiseSchedule
from torch import Tensor
import torch
import numpy as np

class CosineSchedule(NoiseSchedule):
    """
    Cosine beta schedule class for variance preserving diffusion models.

    Args:
        s: Small offset parameter to avoid singularities at t=0 (default: 0.008).

    Methods:
        alphas_cumprod(t): Computes cumulative alpha from t using cosine schedule.
        beta(t): Computes beta at time t using cosine schedule.
    """

    def __init__(self, s: float = 0.008, T: float = 1.0):
        self.s = s
        self.T = T  # Duración del intervalo de integración

    def alphas_cumprod(self, t: Tensor) -> Tensor:

        """
        Computes the cumulative product of alpha values for the given time tensor t.
        Implements:
            ᾱ(t) = cos²((t + s)/(1 + s) * π/2)
        Args:
            t (Tensor): Tensor of time values in [0, T].
        Returns:
            Tensor: Cumulative product of alphas evaluated at t.

        Example:
            >>> schedule = CosineSchedule(s=0.008, T=1.0)
            >>> t = torch.tensor([0.0, 0.5, 1.0])
            >>> schedule.alphas_cumprod(t).round(decimals=4)
            tensor([1.0000, 0.4912, 0.0000])
        """
        
        t_norm = t / self.T
        f = lambda u: torch.cos((u + self.s)/(1 + self.s)*torch.pi/2)**2
        device = t.device
        alpha_bar = f(t_norm) / f(torch.zeros(1, device=device))
        eps = 1e-8
        alpha_bar = torch.clamp(alpha_bar, min=eps)
        return alpha_bar

    def beta(self, t: Tensor) -> Tensor:

        """
        Computes the beta value at time t using the cosine schedule.
        Implements:
            β(t) = (π / (T * (1 + s))) * tan((t + s)/(1 + s) * π/2)
        Args:
            t (Tensor): Tensor of time values in [0, T].
        Returns:
            Tensor: Beta values evaluated at t.

        Example:
            >>> schedule = CosineSchedule(s=0.008, T=1.0)
            >>> t = torch.tensor([0.0, 0.5, 1])
            >>> schedule.beta(t).round(decimals=4)
            tensor([0.0389, 0.9990, 0.0000])
        """

        t = torch.clamp(t, min=0.0, max=self.T)
        t_norm = t / self.T
        theta = (t_norm + self.s) / (1 + self.s) * torch.pi / 2
        beta = (torch.pi / (self.T * (1 + self.s))) * torch.tan(theta)
        return torch.clamp(beta, min=0.0, max=0.999)

    
    def integrated_beta(self, t: Tensor) -> Tensor:

        """
        Approximate the integral of beta(s) ds from 0 to t
        using the trapezoidal rule.

        Args:
            t: Tensor of time values (batch_size,)

        Returns:
            Tensor of shape (batch_size,) with ∫₀ᵗ β(s) ds
        """

        num_points = 500  # número de puntos para integrar
        device = t.device
        result = []

        for t_i in t:
            s_vals = torch.linspace(0, t_i, num_points, device=device)
            beta_vals = self.beta(s_vals)
            dt = t_i / (num_points - 1)
            integral = torch.trapz(beta_vals, dx=dt)
            result.append(integral)

        return torch.stack(result)
    
    def integrated_beta_analytical(self, t: Tensor) -> Tensor:
        return -torch.log(self.alphas_cumprod(t))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
