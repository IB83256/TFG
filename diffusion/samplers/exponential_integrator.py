# -*- coding: utf-8 -*-
"""
Exponential integrator for stochastic differential equations (SDEs)

Author: Álvaro Duro 
Date: 01-09-2026
"""

from typing import Callable, Union
import torch
from torch import Tensor

def exponential_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_linear_coeff: Callable[[Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    score_function: Callable[[Tensor, Tensor], Tensor],
    mask: Union[Tensor, None] = None,
    seed: Union[int, None] = None
) -> tuple[Tensor, Tensor]:
    """
    Exponential integrator for SDEs of the form:
        dx = -A(t) x dt + g(t)^2 ∇_x log p_t(x) dt

    The linear part is integrated exactly via the exponential of A(t),
    and the nonlinear (score-based) part is handled numerically.

    Args:
        x_0: Initial state tensor, shape (batch_size, channels, H, W)
        t_0: Initial time
        t_end: Final time
        n_steps: Number of time steps
        drift_linear_coeff: Function A(t), scalar or (batch_size,) tensor
        diffusion_coefficient: Function g(t), scalar or (batch_size,) tensor
        score_function: Function (x, t) → score estimate
        mask: Optional binary mask for inpainting/imputation
        seed: Optional seed for reproducibility

    Returns:
        times: Discretized time points, shape (n_steps + 1,)
        x_t: Trajectory tensor, shape (*x_0.shape, n_steps + 1)

    Example:
        >>> x_0 = torch.zeros(2, 1, 4, 4)
        >>> A = lambda t: torch.ones_like(t)
        >>> g = lambda t: torch.ones_like(t)
        >>> score_fn = lambda x, t: -x
        >>> times, x_t = exponential_integrator(x_0, 0.0, 1.0, 5, A, g, score_fn, seed=42)
        >>> x_t.shape
        torch.Size([2, 1, 4, 4, 6])
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = x_0.device
    dtype = x_0.dtype

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device, dtype=dtype)
    dt = times[1] - times[0]

    x_t = torch.empty((*x_0.shape, n_steps + 1), dtype=dtype, device=device)

    if mask is None:
        x_t[..., 0] = x_0
    else:
        x_init = torch.randn_like(x_0)
        x_init = x_init * (1 - mask) + x_0 * mask
        x_t[..., 0] = x_init

    for n, t in enumerate(times[:-1]):
        t_tensor = torch.full((x_0.shape[0],), t.item(), device=device, dtype=dtype)

        A_t = drift_linear_coeff(t_tensor).view(-1, 1, 1, 1)
        g_t = diffusion_coefficient(t_tensor).view(-1, 1, 1, 1)
        score = score_function(x_t[..., n], t_tensor)

        decay = torch.exp(-A_t * dt)
        increment = g_t ** 2 * score * (1 - decay) / A_t.clamp(min=1e-6)

        x_next = decay * x_t[..., n] + increment
        if mask is not None:
            x_next = x_next * (1 - mask) + x_0 * mask
        x_t[..., n + 1] = x_next

    return times, x_t

if __name__ == "__main__":
    import doctest
    doctest.testmod()
