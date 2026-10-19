# -*- coding: utf-8 -*-

"""
Training utilities for Energy-Based Models using Persistent Contrastive Divergence.

Author: Álvaro Duro
Date: 01-09-2026
"""


import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from ebm.evaluation import mean_energy
from utils import save_checkpoint, save_results, plot_image_grid, to_01


def _train_one_epoch(
    model,
    data_loader,
    optimizer,
    langevin_sampler,
    replay_buffer,
    device,
    sampler_kwargs,
    hist_it,
    alpha,
    data_noise,
    grad_clip_norm,
    max_iters_per_epoch=None,
):
    """
    Train the Energy-Based Model for one epoch using PCD.

    Args:
        model:
            Energy-Based Model.

        data_loader:
            Training dataloader.

        optimizer:
            Optimizer used to update the model parameters.

        langevin_sampler:
            Langevin dynamics sampling function.

        replay_buffer:
            Persistent replay buffer used by PCD.

        device:
            Device where the model is trained.

        sampler_kwargs:
            Parameters used by the Langevin sampler.

        hist_it:
            Dictionary used to store iteration-level metrics.

        alpha:
            Weight of the energy regularization term.

        data_noise:
            Standard deviation of the noise added to real samples.

        grad_clip_norm:
            Maximum norm used for gradient clipping.

        max_iters_per_epoch:
            Optional maximum number of iterations per epoch.
    """

    model.train()

    for iteration, (x_pos, _) in enumerate(data_loader):

        if (
            max_iters_per_epoch is not None
            and iteration >= max_iters_per_epoch
        ):
            break

        # Move real samples to the selected device
        x_pos = x_pos.to(
            device,
            non_blocking=True,
        )

        # Add small Gaussian noise to real samples
        x_pos = (
            x_pos
            + data_noise * torch.randn_like(x_pos)
        ).clamp(-1, 1)

        # Sample initial negative samples from the replay buffer
        x_init, indices = replay_buffer.sample(
            x_pos.size(0)
        )

        # Generate negative samples using Langevin dynamics
        x_neg = langevin_sampler(
            model=model,
            x_init=x_init,
            **sampler_kwargs,
        )

        # Update persistent chains
        replay_buffer.update(
            x_neg,
            indices,
        )

        # Compute positive and negative energies
        e_pos, e_neg = model(
            torch.cat([x_pos, x_neg])
        ).chunk(2)

        # Contrastive Divergence loss
        loss_cd = (
            e_pos.mean()
            - e_neg.mean()
        )

        # Energy regularization
        loss_reg = alpha * (
            e_pos.pow(2)
            + e_neg.pow(2)
        ).mean()

        # Total loss
        loss = loss_cd + loss_reg

        # Optimization
        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip_norm,
        )

        optimizer.step()

        # Check numerical stability
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss: the model has diverged."
            )

        # Store iteration-level metrics
        hist_it["loss"].append(
            loss.item()
        )

        hist_it["cd"].append(
            loss_cd.item()
        )

        hist_it["reg"].append(
            loss_reg.item()
        )

        hist_it["e_real"].append(
            e_pos.mean().item()
        )

        hist_it["e_fake"].append(
            e_neg.mean().item()
        )


def train_energy_model_pcd(
    model,
    data_loader,
    optimizer,
    scheduler,
    langevin_sampler,
    replay_buffer,
    device,
    sampler_kwargs,
    config,
    hist_it,
    hist_ep,
    x_test_fixed,
    x_noise_fixed,
    gen_init_fixed,
):
    """
    Train an Energy-Based Model using Persistent Contrastive Divergence.

    The function controls the complete training process:
    - PCD training.
    - Learning-rate scheduling.
    - Epoch-level evaluation.
    - Metric history.
    - Divergence warnings.
    - Checkpoints.
    - Periodic sample generation.
    - Visualization.

    Args:
        model:
            Energy-Based Model.

        data_loader:
            Training dataloader.

        optimizer:
            Optimizer used during training.

        scheduler:
            Learning-rate scheduler.

        langevin_sampler:
            Langevin dynamics sampling function.

        replay_buffer:
            Persistent replay buffer.

        device:
            Device where the model is trained.

        sampler_kwargs:
            Parameters used by the Langevin sampler.

        config:
            Experiment configuration dictionary.

        hist_it:
            Dictionary containing iteration-level metrics.

        hist_ep:
            Dictionary containing epoch-level metrics.

        x_test_fixed:
            Fixed real test samples used for evaluation.

        x_noise_fixed:
            Fixed noise samples used for evaluation.

        gen_init_fixed:
            Fixed noise samples used for periodic generation.

    Returns:
        hist_it:
            Iteration-level training history.

        hist_ep:
            Epoch-level training history.
    """

    # Allows training to continue if the cell is executed again
    start_epoch = len(hist_ep["epoch"]) + 1

    # If True, every epoch overwrites a single checkpoint ("last.pt")
    # instead of writing one file per epoch. The replay buffer is then
    # stored as well, so training can be resumed with the persistent
    # chains intact.
    only_last = config.get("only_last", True)

    for epoch in range(
        start_epoch,
        config["epochs"] + 1,
    ):

        t0 = time.time()

        # Train one epoch

        _train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            langevin_sampler=langevin_sampler,
            replay_buffer=replay_buffer,
            device=device,
            sampler_kwargs=sampler_kwargs,
            hist_it=hist_it,
            alpha=config["alpha"],
            data_noise=config["data_noise"],
            grad_clip_norm=config["grad_clip_norm"],
            max_iters_per_epoch=config["max_iters_per_epoch"],
        )

        # Learning rate used during this epoch
        lr_now = optimizer.param_groups[0]["lr"]

        # Update learning rate for the next epoch
        scheduler.step()

        # Epoch training time
        dt = time.time() - t0

        # Epoch-level metrics

        e_test = mean_energy(
            model,
            x_test_fixed,
        )

        e_noise = mean_energy(
            model,
            x_noise_fixed,
        )

        e_buf = mean_energy(
            model,
            replay_buffer.random_view(1024),
        )

        for key, value in zip(
            hist_ep,
            [
                epoch,
                e_test,
                e_noise,
                e_buf,
                lr_now,
                dt,
            ],
        ):
            hist_ep[key].append(value)

        # Divergence warning

        warning = ""

        if e_buf < e_test - 0.5:
            warning = (
                "  <-- AVISO: las cadenas tienen menos energía "
                "que los datos (posible divergencia)"
            )

        print(
            f"época {epoch:3d}/{config['epochs']} | "
            f"{dt:6.1f}s | "
            f"E(test) {e_test:+.3f} | "
            f"E(ruido) {e_noise:+.3f} | "
            f"E(buffer) {e_buf:+.3f} | "
            f"CD {np.mean(hist_it['cd'][-100:]):+.4f}"
            f"{warning}"
        )

        # Epoch checkpoint

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            replay_buffer=replay_buffer,
            epoch=epoch,
            config=config,
            with_buffer=only_last,
            name="last.pt" if only_last else None,
        )

        # Results are saved every epoch so that an interrupted run
        # still leaves a usable training history on disk.

        save_results(
            config=config,
            hist_it=hist_it,
            hist_ep=hist_ep,
        )

        # Periodic visualization

        if (
            epoch % config["vis_every"] == 0
            or epoch == config["epochs"]
        ):

            gen = langevin_sampler(
                model=model,
                x_init=gen_init_fixed,
                n_steps=config["gen_steps"],
                step_size=config["sgld_step_size"],
                noise_scale=config["sgld_noise"],
                grad_clip=config["sgld_grad_clip"],
            )

            images = torch.cat([
                gen,
                replay_buffer.random_view(16),
            ])

            fig, ax = plot_image_grid(
                images=to_01(
                    images.detach().cpu()
                ),
                figsize=(7, 7),
                n_cols=8,
            )

            ax.set_title(
                f"época {epoch} — "
                f"fila 1-2: desde ruido "
                f"({config['gen_steps']} pasos) | "
                f"fila 3-4: buffer"
            )

            plt.show()

    # Final checkpoint including the persistent replay buffer

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        replay_buffer=replay_buffer,
        epoch=config["epochs"],
        config=config,
        with_buffer=True,
        name="last.pt",
    )
    save_results(
        config=config,
        hist_it=hist_it,
        hist_ep=hist_ep,
    )

    return hist_it, hist_ep