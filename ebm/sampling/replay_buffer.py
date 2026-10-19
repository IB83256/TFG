# -*- coding: utf-8 -*-
"""
Persistent replay buffer for PCD training.

Author: Álvaro Duro
Date: 01-09-2026
"""

import torch


class ReplayBuffer:
    """
    Persistent buffer used in Persistent Contrastive Divergence.

    The buffer stores negative samples that are reused as the initial
    state of the Markov chains in subsequent training iterations.
    """

    def __init__(
        self,
        size,
        image_shape,
        reinit_prob,
        device,
    ):
        self.buffer = (
            torch.rand(
                size,
                *image_shape,
                device=device,
            ) * 2 - 1
        )

        self.reinit_prob = reinit_prob

    def sample(self, batch_size):
        """
        Sample images from the persistent buffer.

        A fraction of the samples is randomly reinitialized.
        """

        indices = torch.randint(
            0,
            len(self.buffer),
            (batch_size,),
            device=self.buffer.device,
        )

        samples = self.buffer[indices].clone()

        reinit = (
            torch.rand(
                batch_size,
                device=self.buffer.device,
            )
            < self.reinit_prob
        )

        samples[reinit] = (
            torch.rand_like(samples[reinit]) * 2 - 1
        )

        return samples, indices

    def update(self, samples, indices):
        """
        Update the persistent chains with the new generated samples.
        """

        self.buffer[indices] = samples.detach()

    def random_view(self, n_samples):
        """
        Return random samples from the buffer for visualization
        or evaluation.
        """

        indices = torch.randint(
            0,
            len(self.buffer),
            (n_samples,),
            device=self.buffer.device,
        )

        return self.buffer[indices]