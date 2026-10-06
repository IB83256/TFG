# -*- coding: utf-8 -*-
"""
Noise schedule utilities for NCSNv2-style multi-scale DSM.

Author: Álvaro Duro
"""

import math

import numpy as np
import torch


@torch.no_grad()
def max_pairwise_distance(
    dataset,
    device,
    n=10000,
    chunk=1000,
):
    idx = torch.randperm(
        len(dataset)
    )[:n]

    x = (
        dataset.data[idx]
        .float()
        .div(255.0)
        .view(n, -1)
        .to(device)
    )

    dmax = 0.0
    medians = []

    for i in range(0, n, chunk):
        distances = torch.cdist(
            x[i:i + chunk],
            x,
        )

        dmax = max(
            dmax,
            distances.max().item(),
        )

        random_indices = torch.randint(
            0,
            distances.numel(),
            (5000,),
            device=device,
        )

        medians.append(
            distances.flatten()[
                random_indices
            ]
        )

    dmed = (
        torch.cat(medians)
        .median()
        .item()
    )

    return dmax, dmed


def _normal_cdf(value):
    return 0.5 * (
        1
        + math.erf(
            value / math.sqrt(2)
        )
    )


def noise_overlap(
    gamma,
    dimension,
):
    a = (
        math.sqrt(2 * dimension)
        * (gamma - 1)
    )

    return (
        _normal_cdf(
            a + 3 * gamma
        )
        - _normal_cdf(
            a - 3 * gamma
        )
    )


def _eq4_ratio(
    eps,
    n_steps_per_level,
    gamma,
    sigma_l,
):
    a = eps / sigma_l ** 2
    b = 2.0 / (2.0 - a)

    return (
        (1 - a) ** (
            2 * n_steps_per_level
        )
        * (gamma ** 2 - b)
        + b
    )


def build_noise_schedule(
    dataset,
    image_shape,
    device,
    sigma_1="auto",
    sigma_l=0.01,
    overlap_c=0.5,
    n_steps_per_level=5,
    eps="auto",
):
    """
    Build the geometric noise schedule and Langevin epsilon
    used by the original DSM notebook.
    """

    dimension = int(
        np.prod(image_shape)
    )

    dmax, dmed = max_pairwise_distance(
        dataset=dataset,
        device=device,
    )

    sigma_1_value = (
        dmax
        if sigma_1 == "auto"
        else float(sigma_1)
    )

    sigma_l_value = float(sigma_l)

    lo = 1.0 + 1e-9
    hi = 2.0

    for _ in range(100):
        mid = (lo + hi) / 2

        if (
            noise_overlap(
                mid,
                dimension,
            )
            > overlap_c
        ):
            lo = mid
        else:
            hi = mid

    gamma_target = (
        lo + hi
    ) / 2

    n_levels = (
        int(
            math.ceil(
                math.log(
                    sigma_1_value
                    / sigma_l_value
                )
                / math.log(
                    gamma_target
                )
            )
        )
        + 1
    )

    sigmas = torch.tensor(
        np.geomspace(
            sigma_1_value,
            sigma_l_value,
            n_levels,
        ),
        dtype=torch.float32,
        device=device,
    )

    gamma = (
        sigmas[0]
        / sigmas[1]
    ).item()

    if eps == "auto":
        grid = np.logspace(
            -9,
            math.log10(
                0.5
                * sigma_l_value ** 2
            ),
            20000,
        )

        ratios = np.array([
            _eq4_ratio(
                value,
                n_steps_per_level,
                gamma,
                sigma_l_value,
            )
            for value in grid
        ])

        eps_value = float(
            grid[
                np.argmin(
                    np.abs(
                        ratios - 1
                    )
                )
            ]
        )

    else:
        eps_value = float(eps)

    ratio = _eq4_ratio(
        eps_value,
        n_steps_per_level,
        gamma,
        sigma_l_value,
    )

    nfe = (
        n_levels
        * n_steps_per_level
        + 1
    )

    return {
        "sigmas": sigmas,
        "sigma_1": sigma_1_value,
        "sigma_l": sigma_l_value,
        "gamma": gamma,
        "n_levels": n_levels,
        "eps": eps_value,
        "eq4_ratio": ratio,
        "nfe": nfe,
        "dmax": dmax,
        "dmed": dmed,
        "overlap": noise_overlap(
            gamma,
            dimension,
        ),
    }


def compute_effective_noise(
    score,
    sigma,
):
    """
    Compute the normalized effective noise
    associated with the predicted score.

    Args:
        score: Score predicted by the model.
        sigma: Current noise level.

    Returns:
        Mean normalized effective noise.
    """

    dimension = score[0].numel()

    effective_noise = (
        (
            sigma ** 2
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