# -*- coding: utf-8 -*-
"""
Annealed Langevin sampling for multi-scale score models.

Author: Álvaro Duro
"""

import math

import numpy as np
import torch
from ebm.noise import compute_effective_noise

@torch.no_grad()
def annealed_langevin(
    model,
    sigmas,
    eps,
    image_shape,
    n_steps_per_level,
    device,
    n_samples=16,
    x_init=None,
    seed=None,
    denoise=True,
    save_levels=None,
    batch_size=250,
):
    """
    Generate samples using annealed Langevin dynamics.

    If save_levels is provided, returns:
        samples, snapshots, effective_noise
    """

    generator = (
        torch.Generator(
            device=device
        ).manual_seed(seed)
        if seed is not None
        else None
    )

    def randn(shape):
        return torch.randn(
            shape,
            generator=generator,
            device=device,
        )

    def rand(shape):
        return torch.rand(
            shape,
            generator=generator,
            device=device,
        )

    model.eval()

    outputs = []
    snapshots_all = []
    noise_all = []

    for start in range(
        0,
        n_samples,
        batch_size,
    ):
        batch_size_current = min(
            batch_size,
            n_samples - start,
        )

        if x_init is not None:
            x = (
                x_init[
                    start:
                    start + batch_size_current
                ]
                .to(device)
                .clone()
            )

        else:
            x = (
                rand(
                    (
                        batch_size_current,
                        *image_shape,
                    )
                )
                + sigmas[0]
                * randn(
                    (
                        batch_size_current,
                        *image_shape,
                    )
                )
            )

        snapshots = {}
        effective_noise = []

        for level, sigma in enumerate(
            sigmas
        ):
            alpha = (
                eps
                * (
                    sigma
                    / sigmas[-1]
                ) ** 2
            )

            sigma_batch = sigma.expand(
                batch_size_current
            )

            for _ in range(
                n_steps_per_level
            ):
                score = model(
                    x,
                    sigma_batch,
                )

                x = (
                    x
                    + alpha * score
                    + torch.sqrt(
                        2 * alpha
                    )
                    * randn(x.shape)
                )

            effective_noise.append(
                compute_effective_noise(
                    score,
                    sigma,
                )
            )

            if (
                save_levels is not None
                and level in save_levels
            ):
                snapshots[level] = (
                    x.clamp(0, 1)
                    .cpu()
                )

        if denoise:
            x = (
                x
                + sigmas[-1] ** 2
                * model(
                    x,
                    sigmas[-1].expand(
                        batch_size_current
                    ),
                )
            )

        outputs.append(
            x.clamp(0, 1)
            .cpu()
        )

        snapshots_all.append(
            snapshots
        )

        noise_all.append(
            effective_noise
        )

    samples = torch.cat(
        outputs
    )

    if save_levels is not None:
        snapshots = {
            level: torch.cat([
                batch_snapshots[level]
                for batch_snapshots
                in snapshots_all
            ])
            for level in save_levels
        }

        return (
            samples,
            snapshots,
            np.mean(
                noise_all,
                axis=0,
            ),
        )

    return samples


@torch.no_grad()
def langevin_at_sigma(
    model,
    x,
    sigma,
    n_steps,
    step_size,
    save_steps,
    device,
    seed=0,
):
    """
    Continue a chain at one fixed noise level.
    """

    generator = (
        torch.Generator(
            device=device
        ).manual_seed(seed)
    )

    x = (
        x.to(device)
        .clone()
    )

    sigma_batch = sigma.expand(
        len(x)
    )

    snapshots = {}

    if 0 in save_steps:
        snapshots[0] = (
            x
            + sigma ** 2
            * model(
                x,
                sigma_batch,
            )
        ).clamp(
            0,
            1,
        ).cpu()

    for step in range(
        1,
        n_steps + 1,
    ):
        x = (
            x
            + step_size
            * model(
                x,
                sigma_batch,
            )
            + math.sqrt(
                2 * step_size
            )
            * torch.randn(
                x.shape,
                generator=generator,
                device=device,
            )
        )

        if step in save_steps:
            snapshots[step] = (
                x
                + sigma ** 2
                * model(
                    x,
                    sigma_batch,
                )
            ).clamp(
                0,
                1,
            ).cpu()

    return snapshots