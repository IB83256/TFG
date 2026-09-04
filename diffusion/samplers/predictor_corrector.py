# -*- coding: utf-8 -*-
"""
Predictor-Corrector integrator for stochastic differential equations (SDEs)

Author: Álvaro Duro
Date: 
"""

from typing import Callable, Union
import torch
from torch import Tensor

def predictor_corrector_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    score_function: Callable[[Tensor, Tensor], Tensor],
    n_corrector_steps: int = 1,
    corrector_step_size: float = 0.01,
    mask: Union[Tensor, None] = None,
    seed: Union[int, None] = None
) -> tuple[Tensor, Tensor]:

    """
    Predictor-Corrector integrator for SDEs of the form:
        dx_t = f(x_t, t) dt + g(t) dW_t

    Args:
        x_0: Initial tensor (e.g., images), shape (batch_size, n_channels, H, W)
        t_0: Initial time
        t_end: Final time
        n_steps: Number of discretization steps
        drift_coefficient: Function f(x, t) representing the drift term
        diffusion_coefficient: Function g(t) representing the diffusion term
        score_function: Score-based model providing estimates of gradients of log probability
        n_corrector_steps: Number of corrector (Langevin) steps per time step
        corrector_step_size: Step size for corrector updates
        seed: Optional integer seed for reproducibility

    Returns:
        times: Tensor of discretized time points, shape (n_steps + 1,)
        x_t: Tensor containing the evolution of x over time,
             shape (*x_0.shape, n_steps + 1)

    Example:
        >>> import torch
        >>> x_0 = torch.zeros(2, 1, 4, 4)
        >>> drift = lambda x, t: torch.zeros_like(x)
        >>> diffusion = lambda t: torch.ones_like(t)
        >>> score_fn = lambda x, t: torch.randn_like(x)
        >>> times, x_t = predictor_corrector_integrator(x_0, t_0=0.0, t_end=1.0, n_steps=5, drift_coefficient=drift, diffusion_coefficient=diffusion, score_function=score_fn, seed=42)
        >>> x_t.shape
        torch.Size([2, 1, 4, 4, 6])
    """

    # Set reproducibility if seed is given
    if seed is not None:
        torch.manual_seed(seed)

    device = x_0.device
    dtype = x_0.dtype

    # Create time vector
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device, dtype=dtype)
    dt = times[1] - times[0]
    dt_sqrt = torch.sqrt(torch.abs(dt))

    # Initialize x_t
    x_t = torch.empty((*x_0.shape, n_steps + 1), dtype=dtype, device=device)

    if mask is None:
        x_t[..., 0] = x_0
    else:
        x_init = torch.randn_like(x_0)
        x_init = x_init * (1 - mask) + x_0 * mask
        x_t[..., 0] = x_init


    # Sample all noise terms in advance
    z = torch.randn_like(x_t)

    for n, t in enumerate(times[:-1]):
        t_tensor = torch.full((x_0.shape[0],), t.item(), device=device, dtype=dtype)

        # Predictor step (Euler-Maruyama)
        x_pred = (
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t_tensor) * dt
            + diffusion_coefficient(t_tensor).view(-1, 1, 1, 1) * dt_sqrt * z[..., n]
        )

        # Corrector steps (Langevin dynamics guided by the score)
        x_corr = x_pred
        for _ in range(n_corrector_steps):
            noise = torch.randn_like(x_corr)
            grad_logp = score_function(x_corr, t_tensor)
            x_corr = x_corr + corrector_step_size * grad_logp + torch.sqrt(torch.tensor(2.0 * corrector_step_size, device=device)) * noise

        if mask is not None:
            x_corr = x_corr * (1 - mask) + x_0 * mask

        x_t[..., n + 1] = x_corr

    return times, x_t

if __name__ == "__main__":
    import doctest
    doctest.testmod()
