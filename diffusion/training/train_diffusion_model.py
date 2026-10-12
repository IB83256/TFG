# -*- coding: utf-8 -*-
"""
Training utilities for score-based diffusion models.

Author: Álvaro Duro
"""

import time

import torch

from utils import (
    save_checkpoint,
    save_results,
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
        )

        save_results(
            config=config,
            hist_it=hist_it,
            hist_ep=hist_ep,
        )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config["epochs"],
        config=config,
        name="last.pt",
    )

    return hist_it, hist_ep