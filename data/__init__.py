# -*- coding: utf-8 -*-
"""
Initialization of the data package.

Author: Álvaro Duro 
Date: 01-09-2026
"""

from .mnist import get_mnist_loaders

__all__ = ["get_mnist_loaders"]