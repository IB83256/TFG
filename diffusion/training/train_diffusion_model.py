# -*- coding: utf-8 -*-
"""
Training utilities for score-based diffusion models.

Author: Álvaro Duro
Date: 01-09-2026
"""

import time

import torch
import matplotlib.pyplot as plt

from diffusion.samplers import euler_maruyama_integrator
from utils import (
    save_checkpoint,
    save_results,
    plot_image_grid,
    sample_initial_latents,
)


def _train_one_epoch(
    model,
    diffusion_process,
    data_loader,
    optimizer,
    device,
    hist_it,
    n_time_bins=10,
):
    """
    Train the score model for one epoch.
    """

    model.train()

    total_loss = 0.0
    num_items = 0

    sum_loss_t = torch.zeros(
        n_time_bins,
        device=device,
    )

    count_t = torch.zeros(
        n_time_bins,
        device=device,
    )

    for x, _ in data_loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        loss, per_sample, t = (
            diffusion_process.loss_function(
                model,
                x,
                return_details=True,
            )
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Pérdida no finita durante "
                "el entrenamiento."
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        hist_it["loss"].append(
            loss.item()
        )

        total_loss += (
            loss.item()
            * x.shape[0]
        )

        num_items += x.shape[0]

        # Intervalo temporal de cada muestra
        indices_t = torch.clamp(
            (t * n_time_bins).long(),
            max=n_time_bins - 1,
        )

        sum_loss_t.index_add_(
            0,
            indices_t,
            per_sample,
        )

        count_t.index_add_(
            0,
            indices_t,
            torch.ones_like(
                per_sample
            ),
        )

    epoch_loss = (
        total_loss
        / num_items
    )

    loss_per_t = (
        sum_loss_t
        / count_t.clamp(min=1)
    ).cpu().tolist()

    return (
        epoch_loss,
        loss_per_t,
    )


@torch.no_grad()
def _sample_for_visualization(
    model,
    diffusion_process,
    x_init,
    config,
):
    """
    Generate samples for periodic monitoring.

    Uses a reduced number of integration steps: the goal is to see
    whether the model is learning, not to produce final samples.
    """

    was_training = model.training

    model.eval()

    _, trajectory = euler_maruyama_integrator(
        x_init,
        t_0=config["t_start"],
        t_end=config["t_end"],
        n_steps=config.get(
            "vis_steps",
            200,
        ),
        drift_coefficient=(
            lambda x_t, t:
            diffusion_process.reverse_drift(
                x_t,
                t,
                model,
            )
        ),
        diffusion_coefficient=(
            diffusion_process.diffusion_coefficient
        ),
    )

    model.train(was_training)

    # The integrator returns shape (*x_init.shape, n_steps + 1)
    return trajectory[..., -1]


def train_diffusion_model(
    model,
    diffusion_process,
    data_loader,
    optimizer,
    device,
    config,
    hist_it,
    hist_ep,
):
    """
    Train a score-based diffusion model.
    """

    start_epoch = (
        len(hist_ep["epoch"])
        + 1
    )

    # If True, every epoch overwrites a single checkpoint ("last.pt")
    # instead of writing one file per epoch.
    only_last = config.get("only_last", True)

    vis_every = config.get("vis_every", 5)

    n_vis = config.get("vis_samples", 16)

    # Fixed initial latent, so the samples of different epochs are
    # comparable. The VE prior at t = t_start has standard deviation
    # sigma_t(t_start), not 1.
    generator_state = torch.random.get_rng_state()

    torch.manual_seed(1234)

    sigma_start = diffusion_process.sigma_t(
        torch.tensor(
            [config["t_start"]],
            device=device,
        )
    ).item()

    vis_init = sample_initial_latents(
        n_images=n_vis,
        image_shape=(1, 28, 28),
        mean=0.0,
        std=sigma_start,
        device=device,
    )

    torch.random.set_rng_state(generator_state)

    for epoch in range(
        start_epoch,
        config["epochs"] + 1,
    ):
        t0 = time.time()

        epoch_loss, loss_per_t = (
            _train_one_epoch(
                model=model,
                diffusion_process=diffusion_process,
                data_loader=data_loader,
                optimizer=optimizer,
                device=device,
                hist_it=hist_it,
            )
        )

        elapsed = (
            time.time()
            - t0
        )

        hist_ep["epoch"].append(
            epoch
        )

        hist_ep["loss"].append(
            epoch_loss
        )

        hist_ep["loss_per_t"].append(
            loss_per_t
        )

        hist_ep["time_s"].append(
            elapsed
        )

        print(
            f"época {epoch:3d}/{config['epochs']} | "
            f"{elapsed:6.1f}s | "
            f"loss {epoch_loss:9.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            name="last.pt" if only_last else None,
        )

        save_results(
            config=config,
            hist_it=hist_it,
            hist_ep=hist_ep,
        )

        # Periodic visualization

        if (
            epoch % vis_every == 0
            or epoch == config["epochs"]
        ):
            samples = _sample_for_visualization(
                model=model,
                diffusion_process=diffusion_process,
                x_init=vis_init,
                config=config,
            )

            fig, ax = plot_image_grid(
                images=samples
                .detach()
                .cpu()
                .clamp(0, 1),
                figsize=(7, 7),
                n_cols=8,
            )

            ax.set_title(
                f"época {epoch} — "
                f"{n_vis} muestras "
                f"({config.get('vis_steps', 200)} pasos, "
                "mismo ruido inicial)"
            )

            plt.show()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config["epochs"],
        config=config,
        name="last.pt",
    )

    return hist_it, hist_ep