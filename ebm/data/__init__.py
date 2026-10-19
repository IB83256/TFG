# -*- coding: utf-8 -*-
"""
Implementation of data generation routines for energy-based models.

Author: Álvaro Duro
Date: 01-09-2026
"""



from .eight_gaussians import sample_eight_gaussians, make_grid

__all__ = [
    "sample_eight_gaussians",
    "make_grid",
]