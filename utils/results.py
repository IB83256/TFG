# -*- coding: utf-8 -*-
"""
Utilities for saving experimental results.

Author: Álvaro Duro
"""

import os
import json


def save_results(
    config,
    hist_it,
    hist_ep,
    name="results.json",
):
    """
    Save the results of an experiment.

    Args:
        config:
            Configuration dictionary used in the experiment.

        hist_it:
            Iteration-level training history.

        hist_ep:
            Epoch-level training history.

        name:
            Name of the results file.

    Returns:
        Path of the saved results file.
    """

    # Create the results directory if it does not exist
    os.makedirs(
        config["results_dir"],
        exist_ok=True,
    )

    results = {
        "config": config,
        "hist_it": hist_it,
        "hist_ep": hist_ep,
    }

    path = os.path.join(
        config["results_dir"],
        name,
    )

    with open(path, "w") as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    return path


def load_results(
    config,
    name="results.json",
):
    """
    Load the results of an experiment.

    Args:
        config:
            Configuration dictionary containing the results directory.

        name:
            Name of the results file.

    Returns:
        Dictionary containing the saved results.
    """

    path = os.path.join(
        config["results_dir"],
        name,
    )

    with open(path, "r") as file:
        results = json.load(file)

    return results