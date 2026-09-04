# -*- coding: utf-8 -*-
"""
Deterministic ODE integrator using Euler method for probability flow ODE.

Author: Álvaro Duro 
Date: 
"""

from typing import Callable
import torch
from torch import Tensor

def euler_ode_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Deterministic Euler integrator for ODEs of the form:
        dx/dt = f(x, t)

    Used in probability flow ODE sampling for generative models.

    Args:
        x_0: Initial tensor, shape (batch_size, channels, H, W)
        t_0: Initial time
        t_end: Final time
        n_steps: Number of time steps
        drift_function: Function f(x, t) defining the ODE drift

    Returns:
        times: Discretized time vector (n_steps + 1,)
        x_t: Tensor containing trajectory of x, shape (*x_0.shape, n_steps + 1)

    Example:
        >>> x0 = torch.zeros(2, 1, 4, 4)
        >>> drift = lambda x, t: torch.ones_like(x)
        >>> times, xt = euler_ode_integrator(x0, 0.0, 1.0, 10, drift)
        >>> xt.shape
        torch.Size([2, 1, 4, 4, 11])
    """

    device = x_0.device
    dtype = x_0.dtype

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device, dtype=dtype)
    dt = times[1] - times[0]

    
    # Initialize x_t
    x_t = torch.empty((*x_0.shape, n_steps + 1), dtype=dtype, device=device)

    if mask is None:
        x_t[..., 0] = x_0
    else:
        # Inicializamos con ruido donde no se conoce el valor, y mantenemos x_0 donde sí
        x_init = torch.randn_like(x_0)
        x_init = x_init * (1 - mask) + x_0 * mask
        x_t[..., 0] = x_init


    for n, t in enumerate(times[:-1]):
        t_tensor = torch.full((x_0.shape[0],), t.item(), device=device, dtype=dtype)
        x_next = x_t[..., n] + drift_coefficient(x_t[..., n], t_tensor) * dt
        if mask is not None:
            x_next = x_next * (1 - mask) + x_0 * mask
        x_t[..., n + 1] = x_next

    return times, x_t

if __name__ == "__main__":
    import doctest
    doctest.testmod()
