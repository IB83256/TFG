# -*- coding: utf-8 -*-
"""
Some evaluation metrics for generative models. 

Author: Álvaro Duro
Date: 01-09-2026
"""

from .inception_score import compute_inception_score

__all__ = [
    "compute_inception_score",
]
