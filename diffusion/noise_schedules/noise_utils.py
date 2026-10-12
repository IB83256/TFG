import math
import torch
from torch import Tensor
from typing import Callable, Union

def compute_effective_noise(
    score,
    sigma,
):
    """
    Compute the normalized effective noise
    associated with the predicted score.
    """

    dimension = score[0].numel()

    effective_noise = (
        (
            sigma.view(-1, 1, 1, 1) ** 2
            * score
        )
        .flatten(1)
        .norm(dim=1)
        / (
            sigma
            * math.sqrt(dimension)
        )
    )

    return effective_noise.mean().item()


@torch.no_grad()
def compute_effective_noise_trajectory(
    model,
    diffusion_process,
    times,
    trajectory,
    step_indices=None,
):
    """
    Compute the effective noise along
    a diffusion trajectory.

    Args:
        model: The neural network model.
        diffusion_process: The diffusion process.
        times: The time points.
        trajectory: The diffusion trajectory.
        step_indices: The indices of the steps to compute the effective noise for.

    Returns:
        A tuple of the selected time points and the corresponding effective noises.
    """

    effective_noise = []
    selected_times = []

    if step_indices is None:
        step_indices = range(len(times))

    for step in step_indices:

        t = times[step]

        t_batch = torch.full(
            (trajectory.shape[0],),
            t.item(),
            device=trajectory.device,
            dtype=trajectory.dtype,
        )

        x = trajectory[
            ...,
            step
        ]

        score = model(
            x,
            t_batch,
        )

        sigma = (
            diffusion_process.sigma_t(
                t_batch
            )
        )

        effective_noise.append(
            compute_effective_noise(
                score,
                sigma,
            )
        )

        selected_times.append(
            t.item()
        )

    return (
        selected_times,
        effective_noise,
    )