# -*- coding: utf-8 -*-
"""
Training utilities for multi-scale Denoising Score Matching.

Works with both ScoreUNet and EnergyUNet: the energy model handles its
own autograd inside forward(), so the loss below is unchanged.

Author: Álvaro Duro
Date: 01-09-2026
"""

import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.visualization import plot_image_grid
from utils import save_checkpoint, save_results

def _dsm_loss(
    model,
    x,
    sigmas,
):
    batch_size = x.shape[0]

    indices = torch.randint(
        0,
        len(sigmas),
        (batch_size,),
        device=x.device,
    )

    sigma = sigmas[indices]

    sigma_view = sigma.view(
        -1,
        1,
        1,
        1,
    )

    noise = torch.randn_like(x)

    score = model(
        x + sigma_view * noise,
        sigma,
    )

    per_sample = (
        (
            sigma_view * score
            + noise
        ) ** 2
    ).sum(
        dim=(1, 2, 3)
    )

    return (
        per_sample.mean(),
        per_sample.detach(),
        indices,
    )


@torch.no_grad()
def _ema_update(
    ema,
    model,
    decay,
):
    for ema_parameter, parameter in zip(
        ema.parameters(),
        model.parameters(),
    ):
        ema_parameter.lerp_(
            parameter,
            1 - decay,
        )

    for ema_buffer, buffer in zip(
        ema.buffers(),
        model.buffers(),
    ):
        ema_buffer.copy_(
            buffer
        )

def _train_one_epoch(
    model,
    ema,
    data_loader,
    optimizer,
    sigmas,
    device,
    config,
    hist_it,
):
    model.train()

    n_levels = len(sigmas)

    sum_loss = torch.zeros(
        n_levels,
        device=device,
    )

    count = torch.zeros(
        n_levels,
        device=device,
    )

    total_loss = 0.0
    n_iterations = 0

    for iteration, (x, _) in enumerate(
        data_loader
    ):
        if (
            config["max_iters_per_epoch"]
            and iteration
            >= config["max_iters_per_epoch"]
        ):
            break

        x = x.to(
            device,
            non_blocking=True,
        )

        loss, per_sample, indices = _dsm_loss(
            model=model,
            x=x,
            sigmas=sigmas,
        )

        # Antes de backward() y step(): si la pérdida no es finita, un
        # paso del optimizador propagaría los NaN a los pesos y, a
        # través de _ema_update, también a la EMA.
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Pérdida no finita. "
                "Recarga un checkpoint anterior."
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["grad_clip"],
        )

        optimizer.step()

        _ema_update(
            ema=ema,
            model=model,
            decay=config["ema_decay"],
        )

        sum_loss.index_add_(
            0,
            indices,
            per_sample,
        )

        count.index_add_(
            0,
            indices,
            torch.ones_like(
                per_sample
            ),
        )

        hist_it["loss"].append(
            loss.item()
        )

        total_loss += loss.item()
        n_iterations += 1

    epoch_loss = (
        total_loss
        / n_iterations
    )

    loss_per_sigma = (
        sum_loss
        / count.clamp(min=1)
    ).cpu().tolist()

    return (
        epoch_loss,
        loss_per_sigma,
    )


def train_energy_model_dsm(
    model,
    ema,
    data_loader,
    optimizer,
    annealed_sampler,
    sigmas,
    eps,
    device,
    sampler_kwargs,
    config,
    hist_it,
    hist_ep,
    vis_init,
):
    """
    Train the score model with multi-scale DSM.

    The training procedure is the same as in the original
    notebook, but the implementation is moved out of the
    notebook to keep the experiment code clean.
    """

    start_epoch = (
        len(
            hist_ep["epoch"]
        )
        + 1
    )

    # If True, every epoch overwrites a single checkpoint ("last.pt")
    # instead of writing one file per epoch.
    only_last = config.get("only_last", True)

    n_levels = len(sigmas)

    for epoch in range(
        start_epoch,
        config["epochs"] + 1,
    ):
        t0 = time.time()

        epoch_loss, loss_per_sigma = (
            _train_one_epoch(
                model=model,
                ema=ema,
                data_loader=data_loader,
                optimizer=optimizer,
                sigmas=sigmas,
                device=device,
                config=config,
                hist_it=hist_it,
            )
        )

        elapsed = (
            time.time()
            - t0
        )

        for key, value in zip(
            hist_ep,
            [
                epoch,
                epoch_loss,
                loss_per_sigma,
                elapsed,
            ],
        ):
            hist_ep[key].append(
                value
            )

        print(
            f"época {epoch:3d}/{config['epochs']} | "
            f"{elapsed:6.1f}s | "
            f"loss {epoch_loss:9.4f} | "
            f"loss(σ grande) "
            f"{np.mean(loss_per_sigma[:n_levels // 3]):8.4f} | "
            f"loss(σ pequeño) "
            f"{np.mean(loss_per_sigma[-n_levels // 3:]):8.4f}"
        )

        save_checkpoint(
            model=model,
            ema=ema,
            optimizer=optimizer,
            sigmas=sigmas,
            eps=eps,
            epoch=epoch,
            config=config,
            name="last.pt" if only_last else None,
        )

        # Results are saved every epoch so that an interrupted run
        # still leaves a usable training history on disk.

        save_results(
            config=config,
            hist_it=hist_it,
            hist_ep=hist_ep,
        )

        if (
            epoch
            % config["vis_every"]
            == 0
            or epoch
            == config["epochs"]
        ):
            samples = annealed_sampler(
                model=ema,
                n_samples=16,
                x_init=vis_init,
                seed=1234,
                **sampler_kwargs,
            )

            fig, ax = plot_image_grid(
                images=samples,
                figsize=(7, 7),
                n_cols=8,
            )

            ax.set_title(
                f"época {epoch} — "
                "16 muestras "
                "(EMA, mismo ruido inicial)"
            )

            plt.show()

    save_checkpoint(
        model=model,
        ema=ema,
        optimizer=optimizer,
        sigmas=sigmas,
        eps=eps,
        epoch=config["epochs"],
        config=config,
        name="last.pt",
    )

    save_results(
        config=config,
        hist_it=hist_it,
        hist_ep=hist_ep,
    )

    return hist_it, hist_ep