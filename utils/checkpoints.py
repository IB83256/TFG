# -*- coding: utf-8 -*-
"""
Checkpoint utilities for Energy-Based Models.

Author: Álvaro Duro
"""

import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    config,
    scheduler=None,
    replay_buffer=None,
    ema=None,
    sigmas=None,
    eps=None,
    with_buffer=False,
    name=None,
):
    """
    Save the current training state.
    """

    os.makedirs(
        config["ckpt_dir"],
        exist_ok=True,
    )

    state = {
        "epoch": epoch,
        "cfg": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    if (
        with_buffer
        and replay_buffer is not None
    ):
        state["buffer"] = (
            replay_buffer.buffer.cpu()
        )

    if ema is not None:
        state["ema"] = ema.state_dict()

    if sigmas is not None:
        state["sigmas"] = sigmas.cpu()

    if eps is not None:
        state["eps"] = eps

    path = os.path.join(
        config["ckpt_dir"],
        name or f"epoch_{epoch:03d}.pt",
    )

    torch.save(
        state,
        path,
    )

    return path


def load_checkpoint(
    model,
    optimizer,
    device,
    config,
    scheduler=None,
    replay_buffer=None,
    ema=None,
    epoch=None,
    name=None,
):
    """
    Load a previously saved training checkpoint.
    """

    path = os.path.join(
        config["ckpt_dir"],
        name or f"epoch_{epoch:03d}.pt",
    )

    state = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        state["model"]
    )

    optimizer.load_state_dict(
        state["optimizer"]
    )

    if (
        scheduler is not None
        and "scheduler" in state
    ):
        scheduler.load_state_dict(
            state["scheduler"]
        )

    if (
        replay_buffer is not None
        and "buffer" in state
    ):
        replay_buffer.buffer = (
            state["buffer"].to(device)
        )

    if (
        ema is not None
        and "ema" in state
    ):
        ema.load_state_dict(
            state["ema"]
        )

    print(
        "cargado",
        path,
    )

    return state["epoch"]