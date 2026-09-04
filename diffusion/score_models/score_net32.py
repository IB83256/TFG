# -*- coding: utf-8 -*-
"""
ScoreNet32: U-Net-based score model for 32x32 images in diffusion-based generative models.

Author: Álvaro Duro 
Date: 
"""

import torch
import torch.nn as nn
import numpy as np

class GaussianRandomFourierFeatures(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.rff_weights = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.rff_weights[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]

class ScoreNet32(nn.Module):
    def __init__(self, marginal_prob_std, in_channels=1, channels=[32, 64, 128, 256], embed_dim=256):
        super().__init__()
        self.embed = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoder
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 4, stride=2, padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2] * 2, channels[1], 4, stride=2, padding=1, bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1] * 2, channels[0], 4, stride=2, padding=1, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] * 2, in_channels, 3, stride=1, padding=1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))

        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))

        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))

        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        h = self.tconv4(h4) + self.dense5(embed)
        h = self.act(self.tgnorm4(h))

        h = self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))

        h = self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)
        h = self.act(self.tgnorm2(h))

        h = self.tconv1(torch.cat([h, h1], dim=1))
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
