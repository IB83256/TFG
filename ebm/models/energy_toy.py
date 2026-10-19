# -*- coding: utf-8 -*-
"""
Energy model for a simple toy example.

Author: Álvaro Duro
Date: 01-09-2026
"""


import torch.nn as nn

class EnergyModelToy(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class EnergyModelToy1(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)