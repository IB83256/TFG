# -*- coding: utf-8 -*-
"""
Data generation for the eight gaussians dataset.

Author: Álvaro Duro
Date: 01-09-2026
"""


import math
import torch


def sample_eight_gaussians(
    n_samples: int,
    radius: float = 2.0,
    std: float = 0.15,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Genera muestras 2D procedentes de una mezcla de 8 gaussianas
    colocadas uniformemente sobre una circunferencia.

    Parameters
    ----------
    n_samples:
        Número total de puntos a generar.

    radius:
        Radio de la circunferencia sobre la que se colocan
        los centros de las gaussianas.

    std:
        Desviación estándar de cada gaussiana.

    seed:
        Semilla aleatoria opcional para reproducibilidad.

    Returns
    -------
    torch.Tensor
        Tensor de forma (n_samples, 2).
    """

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    angles = torch.linspace(
        0,
        2 * math.pi,
        steps=9,
    )[:-1]

    centers = torch.stack(
        [
            radius * torch.cos(angles),
            radius * torch.sin(angles),
        ],
        dim=1,
    )

    component_indices = torch.randint(
        low=0,
        high=len(centers),
        size=(n_samples,),
        generator=generator,
    )

    selected_centers = centers[component_indices]

    noise = std * torch.randn(
        n_samples,
        2,
        generator=generator,
    )

    samples = selected_centers + noise

    return samples



def make_grid(
    x_min: float = -4.0,
    x_max: float = 4.0,
    y_min: float = -4.0,
    y_max: float = 4.0,
    resolution: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Crea una rejilla 2D para evaluar y visualizar una función de energía.

    Returns
    -------
    points:
        Tensor de forma (resolution * resolution, 2).

    xx:
        Coordenadas X de la rejilla.

    yy:
        Coordenadas Y de la rejilla.
    """

    x = torch.linspace(x_min, x_max, resolution)
    y = torch.linspace(y_min, y_max, resolution)

    xx, yy = torch.meshgrid(
        x,
        y,
        indexing="xy",
    )

    points = torch.stack(
        [
            xx.reshape(-1),
            yy.reshape(-1),
        ],
        dim=1,
    )

    return points, xx, yy