# -*- coding: utf-8 -*-
"""
Langevin dynamics sampling for Energy-Based Models.

Author: Álvaro Duro
Date: 01-09-2026
"""

import torch


def langevin_sampling(
    model,
    x_init,
    n_steps,
    step_size,
    noise_scale,
    grad_clip=None,
    save_steps=None,
):
    """
    Generate samples using Langevin dynamics.

    Args:
        model: Energy-based model to sample from.
        x_init: Initial samples (torch.Tensor).
        n_steps: Number of Langevin steps.
        step_size: Step size for the gradient update.
        noise_scale: Standard deviation of the Gaussian noise added at each step.
        grad_clip: Optional gradient clipping value (default: None).
        save_steps: Optional list of steps to save intermediate results (default: None).
    
    Returns:
        If save_steps is None:
            final_samples: Samples after n_steps of Langevin dynamics.
        If save_steps is not None:
            final_samples: Samples after n_steps of Langevin dynamics.
            snapshots: Dictionary of intermediate samples at specified steps.
            energies: List of mean energy values at each step.
    """

    # Save the original state of the model
    was_training = model.training

    model.eval()

    # Store the original requires_grad state of each parameter
    requires_grad = [
        parameter.requires_grad
        for parameter in model.parameters()
    ]

    # Disable gradients with respect to model parameters.
    # Only the gradient with respect to the input is required.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Copy the initial samples
    x = x_init.detach().clone()

    # Steps that will be stored during sampling
    save_steps = (
        set(save_steps)
        if save_steps is not None
        else None
    )

    snapshots = {}
    energies = []

    if save_steps is not None and 0 in save_steps:
        snapshots[0] = x.cpu()

    # Langevin dynamics
    with torch.enable_grad():

        for step in range(1, n_steps + 1):

            # Add Gaussian noise
            x = (
                x
                + noise_scale * torch.randn_like(x)
            ).clamp(-1, 1)

            x.requires_grad_(True)

            # Compute the energy
            energy = model(x)

            # Gradient of the energy with respect to the input
            gradient = torch.autograd.grad(
                energy.sum(),
                x,
            )[0]

            # Optional gradient clipping
            if grad_clip is not None:
                gradient = gradient.clamp(
                    -grad_clip,
                    grad_clip,
                )

            # Move samples towards regions of lower energy
            x = (
                x
                - step_size * gradient
            ).clamp(-1, 1).detach()

            # Save intermediate results if requested
            if save_steps is not None:

                energies.append(
                    energy.mean().item()
                )

                if step in save_steps:
                    snapshots[step] = x.cpu()

    # Restore the original requires_grad state
    for parameter, state in zip(
        model.parameters(),
        requires_grad,
    ):
        parameter.requires_grad_(state)

    # Restore the previous train/eval state
    model.train(was_training)

    if save_steps is not None:
        return x, snapshots, energies

    return x