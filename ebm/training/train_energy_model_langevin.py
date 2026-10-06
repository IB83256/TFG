from typing import Callable

import torch
from torch import nn
from tqdm import trange


def train_energy_model_langevin(
    model: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    langevin_sampler: Callable,
    n_epochs: int,
    device: torch.device,
    sampler_kwargs: dict,
    energy_reg: float = 1e-3,
):
    """
    Train an Energy-Based Model using short-run Langevin negative samples.
    """

    model.train()

    losses = []

    tqdm_epoch = trange(n_epochs)

    for epoch in tqdm_epoch:

        epoch_loss = 0.0
        n_samples = 0

        for batch_idx, (x_real, _) in enumerate(data_loader):

            x_real = x_real.to(device)

            # Initial negative samples in the same range as MNIST
            x_init = torch.rand_like(x_real)

            # Generate negative samples
            x_negative = langevin_sampler(
                model=model,
                x_init=x_init,
                **sampler_kwargs,
            ).detach()

            # Energies
            energy_real = model(x_real)
            energy_negative = model(x_negative)

            mean_real = energy_real.mean()
            mean_negative = energy_negative.mean()

            # Contrastive objective
            contrastive_loss = (
                mean_real
                - mean_negative
            )

            # Energy magnitude regularization
            regularization_loss = energy_reg * (
                energy_real.pow(2).mean()
                + energy_negative.pow(2).mean()
            )

            loss = (
                contrastive_loss
                + regularization_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if batch_idx % 50 == 0:

                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_idx} | "
                    f"E_real: {mean_real.item():.4e} | "
                    f"E_negative: {mean_negative.item():.4e} | "
                    f"Contrastive: {contrastive_loss.item():.4e} | "
                    f"Reg: {regularization_loss.item():.4e} | "
                    f"Loss: {loss.item():.4e}"
                )

            batch_size = x_real.shape[0]

            epoch_loss += (
                loss.item()
                * batch_size
            )

            n_samples += batch_size

        average_loss = (
            epoch_loss
            / n_samples
        )

        losses.append(
            average_loss
        )

        tqdm_epoch.set_postfix(
            loss=f"{average_loss:.6f}"
        )

    return losses