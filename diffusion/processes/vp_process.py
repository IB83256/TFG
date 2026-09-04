# -*- coding: utf-8 -*-
"""
Variance Preserving (VP) diffusion process.

Author: Álvaro Duro 
Date:
"""

import torch
from torch import Tensor
import numpy as np
from .gaussian_process import GaussianDiffusionProcess
from noise_schedules.base_schedule import NoiseSchedule

class VPProcess(GaussianDiffusionProcess):
    """
    Variance Preserving (VP) diffusion process.

    Args:
        noise_schedule: Instance of a NoiseSchedule (e.g., LinearSchedule, CosineSchedule).
        use_precomputed_sigma: If True, use interpolated sigma_t from internal table.

    Example:
        >>> from noise_schedules.linear import LinearSchedule
        >>> schedule = LinearSchedule(beta_min=0.1, beta_max=20.0)
        >>> process = VPProcess(noise_schedule=schedule)
        >>> x = torch.randn(2, 1, 4, 4)
        >>> t = torch.tensor([0.5, 0.5])
        >>> drift = process.drift_coefficient(x, t)
        >>> diffusion = process.diffusion_coefficient(t)
        >>> drift.shape
        torch.Size([2, 1, 4, 4])
        >>> diffusion.shape
        torch.Size([2])
    """

    def __init__(self, noise_schedule: NoiseSchedule, use_precomputed_sigma: bool = False):
        self.noise_schedule = noise_schedule
        self.use_precomputed_sigma = use_precomputed_sigma

        drift_coefficient = lambda x_t, t: -0.5 * self.beta(t).view(-1, 1, 1, 1) * x_t
        diffusion_coefficient = lambda t: torch.sqrt(self.beta(t))

        def mu_t(x_0, t):

            """
            Mean function for the reverse-time SDE.
            Args:
                x_0: Initial state.
                t: Time variable.
            Returns:
                Mean of the reverse-time SDE.
            """

            return x_0 * torch.sqrt(self.noise_schedule.alphas_cumprod(t))

        if use_precomputed_sigma:
            self._t_vals, self._sigma_vals = self._generate_sigma_table(self.noise_schedule)

            def sigma_t(t):

                """
                Sigma function for the reverse-time SDE.
                Args:
                    t: Time variable.
                Returns:
                    Sigma of the reverse-time SDE.
                """

                t_cpu = t.detach().cpu().numpy()
                interpolated = np.interp(t_cpu, self._t_vals, self._sigma_vals)
                return torch.tensor(interpolated, dtype=torch.float32, device=t.device)
            
        else:
            def sigma_t(t):

                """
                Sigma function for the reverse-time SDE.
                Args:
                    t: Time variable.
                Returns:
                    Sigma of the reverse-time SDE.
                """

                return torch.sqrt(1.0 - self.noise_schedule.alphas_cumprod(t))

        super().__init__(
            drift_coefficient=drift_coefficient,
            diffusion_coefficient=diffusion_coefficient,
            mu_t=mu_t,
            sigma_t=sigma_t,
        )

    def beta(self, t: Tensor) -> Tensor:
        
        """
        Beta(t) computed from the noise schedule.

        Returns:
            Beta(t) values.
        """

        return self.noise_schedule.beta(t)

    def _generate_sigma_table(self, schedule, T=1.0, num_points=1000):

        """
        Generate a table of sigma values for the reverse-time SDE.
        Args:
            schedule: Noise schedule instance.
            T: Maximum time value.
            num_points: Number of points in the table.
        Returns:
            t_vals: Time values.
            sigma_vals: Corresponding sigma values.
        """

        t_vals = torch.linspace(0, T, num_points)
        sigma_vals = []

        for t_i in t_vals:
            s_vals = torch.linspace(0, t_i.item(), num_points)
            ds = t_i.item() / (num_points - 1)
            beta_s = self.beta(s_vals)

            try:
                int_beta_t = schedule.integrated_beta(torch.tensor([t_i]))[0]
                int_beta_s = schedule.integrated_beta(s_vals)
                inner_integrals = int_beta_t - int_beta_s
            except NotImplementedError:
                inner_integrals = []
                for s_j in s_vals:
                    u_vals = torch.linspace(s_j.item(), t_i.item(), num_points)
                    du = (t_i.item() - s_j.item()) / (num_points - 1)
                    beta_u = self.beta(u_vals)
                    int_beta = torch.trapz(beta_u, dx=du)
                    inner_integrals.append(int_beta.item())
                inner_integrals = torch.tensor(inner_integrals)

            integrand = torch.exp(-0.5 * inner_integrals) * beta_s
            sigma2 = torch.trapz(integrand, dx=ds).item()
            sigma_val = np.sqrt(max(sigma2, 1e-8))
            sigma_vals.append(sigma_val)

        t_vals_np = t_vals.numpy()
        sigma_vals_np = np.clip(np.array(sigma_vals), 1e-5, None)
        return t_vals_np, sigma_vals_np

if __name__ == "__main__":
    import doctest
    doctest.testmod()
