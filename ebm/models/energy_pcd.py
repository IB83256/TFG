from torch import nn
import torch

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EnergyModelPCD(nn.Module):
    def __init__(self, hidden=32, spectral_norm=False):
        super().__init__()
        c1, c2, c3 = hidden // 2, hidden, hidden * 2
        sn = nn.utils.parametrizations.spectral_norm if spectral_norm else (lambda m: m)
        self.net = nn.Sequential(
            sn(nn.Conv2d(1,  c1, 5, stride=2, padding=4)), Swish(),   # 16x16
            sn(nn.Conv2d(c1, c2, 3, stride=2, padding=1)), Swish(),   # 8x8
            sn(nn.Conv2d(c2, c3, 3, stride=2, padding=1)), Swish(),   # 4x4
            sn(nn.Conv2d(c3, c3, 3, stride=2, padding=1)), Swish(),   # 2x2
            nn.Flatten(),
            sn(nn.Linear(c3 * 4, c3)), Swish(),
            sn(nn.Linear(c3, 1)),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)      # E_theta(x), shape (B,)