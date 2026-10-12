# -*- coding: utf-8 -*-
"""
Gaussian diffusion process.

Author: Álvaro Duro 
Date: 
"""

import torch
from torch import Tensor
from typing import Callable
from .base_process import DiffusionProcess

class GaussianDiffusionProcess(DiffusionProcess):

    """
    Gaussian Diffusion Process.

    Models both the forward SDE (adding noise) and the reverse-time SDE (sampling by removing noise).
    """

    def __init__(
        self,
        drift_coefficient: Callable[[Tensor, Tensor], Tensor] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[[Tensor], Tensor] = lambda t: 1.0,
        mu_t: Callable[[Tensor, Tensor], Tensor] = lambda x_0, t: x_0,
        sigma_t: Callable[[Tensor], Tensor] = lambda t: torch.sqrt(t),
    ):
        super().__init__(drift_coefficient, diffusion_coefficient)
        self.mu_t = mu_t
        self.sigma_t = sigma_t
    
    def loss_function(
            self,
            score_model, 
            x_0: torch.Tensor, 
            eps: float = 1.0e-5,
            return_details=False,
        ):
            """The loss function for training score-based generative models.

            Args:
                score_model:  A PyTorch model instance that represents a 
                                time-dependent score-based model.
            x_0: A mini-batch of training data.    
            eps: A tolerance value for numerical stability.
            """
            
            # Número de muestras en el batch
            batch_size = x_0.shape[0]
            
            # (1) Muestreamos tiempos uniformemente en (0, 1)
            t = torch.rand(batch_size, device=x_0.device) * (1.0 - eps) + eps
        
            # (2) Muestreamos ruido Gaussiano Z ~ N(0, I)
            z = torch.randn_like(x_0)
        
            # (3) Calculamos sigma(t)
            sigma = self.sigma_t(t).view(-1,1,1,1)
        
            # (4) Generamos x(t) = x_0 + sigma(t) * Z
            x_t = x_0 + sigma * z
        
            # (5) Score predicho por la red
            score = score_model(x_t, t)
        
            # (6) MSE por muestra
            per_sample = torch.norm(
                (score * sigma + z),
                dim=(1, 2, 3),
                p=2,
            )**2

            # (7) MSE 
            loss = torch.mean(per_sample)

            if return_details:
                return loss, per_sample.detach(), t.detach()

            return loss
    
    def loss_function_conditional(self, score_model, x_0, y, eps: float = 1.0e-5) -> torch.Tensor:

        """
        The loss function for training conditional score-based generative models.
        Args:
            score_model: A PyTorch model instance that represents a 
                            time-dependent score-based model.
            x_0: A mini-batch of training data.
            y: Conditional input (e.g., class labels).
            eps: A tolerance value for numerical stability.
        """

        batch_size = x_0.shape[0]
        t = torch.rand(batch_size, device=x_0.device) * (1.0 - eps) + eps
        z = torch.randn_like(x_0)
        sigma = self.sigma_t(t).view(-1, 1, 1, 1)
        x_t = x_0 + sigma * z
        score = score_model(x_t, t, y)
        loss = torch.mean(torch.norm((score * sigma + z), dim=(1, 2, 3), p=2) ** 2)
        return loss


    def reverse_drift(self, x_t: Tensor, t: Tensor, score_model) -> Tensor:

        """
        Computes the reverse-time drift:
            f_reverse(x, t) = f(x, t) - g(t)^2 * score_model(x, t)

        Args:
            x_t: Current state tensor.
            t: Current time tensor.
            score_model: Function (x_t, t) -> estimated score.

        Returns:
            Reverse drift tensor.
        """
        
        f_forward = self.drift_coefficient(x_t, t)
        g = self.diffusion_coefficient(t).view(-1, 1, 1, 1)  
        score = score_model(x_t, t)
        return f_forward - g**2 * score
