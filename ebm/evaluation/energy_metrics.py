# -*- coding: utf-8 -*-
"""
Evaluation utilities for Energy-Based Models.

Author: Álvaro Duro
"""

import torch


@torch.no_grad()
def mean_energy(
    model,
    x,
    batch_size=512,
):
    """
    Compute the mean energy of a set of samples.

    Args:
        model:
            Energy model E(x).

        x:
            Samples whose energy will be evaluated.

        batch_size:
            Batch size used to evaluate the samples.

    Returns:
        Mean energy of the samples.
    """

    # Save the current state of the model
    was_training = model.training

    model.eval()

    # Compute energies in batches
    energy = torch.cat([
        model(
            x[i:i + batch_size]
        )
        for i in range(
            0,
            len(x),
            batch_size,
        )
    ]).mean().item()

    # Restore the previous state of the model
    model.train(was_training)

    return energy