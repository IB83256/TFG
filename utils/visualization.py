# -*- coding: utf-8 -*-
"""
Visualization utilities for image generation and diffusion models.

Functions included:
- plot_image_grid: visualize batches of images in a grid.
- plot_image_evolution: visualize evolution of multiple images over time steps.
- animation_images: create an animation from the temporal evolution of a single image.

Author: Álvaro Duro 
Date: 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
from torchvision.utils import make_grid


def show_tensor_image(ax, image):
    """
    Display a tensor image on a matplotlib axis.

    Supports:
    - Grayscale images with shape (1, H, W)
    - RGB images with shape (3, H, W)
    """

    image = image.clamp(0, 1)

    if image.shape[0] == 1:
        ax.imshow(
            image.squeeze(0),
            cmap="gray",
        )
    else:
        ax.imshow(
            image.permute(1, 2, 0)
        )

def plot_image_grid(
    images: torch.Tensor,
    figsize: tuple,
    n_cols: int,
    padding: int = 2,
    pad_value: float = 1.0,
    normalize: bool = False,
    axis_on_off: str = "off",
):
    """
    Display a batch of images in a grid.
    """

    grid = make_grid(
        images,
        nrow=n_cols,
        padding=padding,
        normalize=normalize,
        pad_value=pad_value,
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    show_tensor_image(
        ax,
        grid,
    )

    ax.axis(axis_on_off)

    return fig, ax


def plot_image_evolution(
    images: torch.Tensor,
    n_images: int,
    n_intermediate_steps,
    figsize: tuple,
):
    """
    Display the temporal evolution of several images.

    Expected shape:
        (N, C, H, W, T)
    """

    fig, axs = plt.subplots(
        n_images,
        len(n_intermediate_steps),
        figsize=figsize,
    )

    for n_image in range(n_images):

        for i, ax in enumerate(
            axs[n_image, :]
        ):

            step = n_intermediate_steps[i]

            image = images[
                n_image,
                :,
                :,
                :,
                step,
            ]

            show_tensor_image(
                ax,
                image,
            )

            ax.set_axis_off()

    return fig, axs


def animation_images(
    images_t,
    interval: int,
    figsize: tuple,
):
    """
    Create an animation from the temporal evolution
    of a single image.

    Expected shapes:
    - Grayscale: (H, W, T)
    - RGB:       (H, W, C, T)
    """

    # Clamp values only for visualization
    images_t = images_t.clamp(0, 1)

    n_frames = images_t.shape[-1]

    fig, ax = plt.subplots(
        figsize=figsize
    )

    if images_t.ndim == 3:

        # Grayscale: (H, W, T)
        img_display = ax.imshow(
            images_t[:, :, 0],
            cmap="gray",
        )

        def update(t):
            img_display.set_array(
                images_t[:, :, t]
            )

            return [img_display]

    else:

        # RGB: (H, W, C, T)
        img_display = ax.imshow(
            images_t[:, :, :, 0]
        )

        def update(t):
            img_display.set_array(
                images_t[:, :, :, t]
            )

            return [img_display]

    ax.axis("off")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=False,
    )

    return fig, ax, anim