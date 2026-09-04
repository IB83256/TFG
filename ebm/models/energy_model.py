import torch.nn as nn


class EnergyModel(nn.Module):
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