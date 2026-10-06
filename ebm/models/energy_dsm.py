# -*- coding: utf-8 -*-
"""
Score network for multi-scale Denoising Score Matching on MNIST.

Author: Álvaro Duro
"""

import math

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    def __init__(self, dim=64, scale=16.0):
        super().__init__()

        self.register_buffer(
            "W",
            torch.randn(dim // 2) * scale,
        )

    def forward(self, t):
        proj = (
            2
            * math.pi
            * t[:, None]
            * self.W[None, :]
        )

        return torch.cat(
            [proj.sin(), proj.cos()],
            dim=-1,
        )


class ResBlock(nn.Module):
    def __init__(
        self,
        cin,
        cout,
        emb_dim,
        groups=8,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            groups,
            cin,
        )

        self.conv1 = nn.Conv2d(
            cin,
            cout,
            3,
            padding=1,
        )

        self.emb = nn.Linear(
            emb_dim,
            cout,
        )

        self.norm2 = nn.GroupNorm(
            groups,
            cout,
        )

        self.conv2 = nn.Conv2d(
            cout,
            cout,
            3,
            padding=1,
        )

        self.skip = (
            nn.Conv2d(cin, cout, 1)
            if cin != cout
            else nn.Identity()
        )

        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.conv1(
            self.act(
                self.norm1(x)
            )
        )

        h = (
            h
            + self.emb(emb)[:, :, None, None]
        )

        h = self.conv2(
            self.act(
                self.norm2(h)
            )
        )

        return h + self.skip(x)


class ScoreUNet(nn.Module):
    def __init__(
        self,
        channels=64,
        emb_dim=128,
        sigma_data=0.5,
    ):
        super().__init__()

        c = channels

        self.sigma_data = sigma_data

        self.embed = nn.Sequential(
            FourierEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.conv_in = nn.Conv2d(
            1,
            c,
            3,
            padding=1,
        )

        self.enc1 = ResBlock(
            c,
            c,
            emb_dim,
        )

        self.down1 = nn.Conv2d(
            c,
            c,
            4,
            stride=2,
            padding=1,
        )

        self.enc2 = ResBlock(
            c,
            2 * c,
            emb_dim,
        )

        self.down2 = nn.Conv2d(
            2 * c,
            2 * c,
            4,
            stride=2,
            padding=1,
        )

        self.mid1 = ResBlock(
            2 * c,
            2 * c,
            emb_dim,
        )

        self.mid2 = ResBlock(
            2 * c,
            2 * c,
            emb_dim,
        )

        self.up2 = nn.ConvTranspose2d(
            2 * c,
            2 * c,
            4,
            stride=2,
            padding=1,
        )

        self.dec2 = ResBlock(
            4 * c,
            c,
            emb_dim,
        )

        self.up1 = nn.ConvTranspose2d(
            c,
            c,
            4,
            stride=2,
            padding=1,
        )

        self.dec1 = ResBlock(
            2 * c,
            c,
            emb_dim,
        )

        self.out = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(
                c,
                1,
                3,
                padding=1,
            ),
        )

        nn.init.zeros_(
            self.out[-1].weight
        )

        nn.init.zeros_(
            self.out[-1].bias
        )

    def forward(self, x, sigma):
        # x: (B, 1, 28, 28)
        # sigma: (B,)

        s = sigma.view(
            -1,
            1,
            1,
            1,
        )

        x_in = (
            (x - 0.5)
            / torch.sqrt(
                s ** 2
                + self.sigma_data ** 2
            )
        )

        emb = self.embed(
            torch.log(sigma) / 4
        )

        e1 = self.enc1(
            self.conv_in(x_in),
            emb,
        )

        e2 = self.enc2(
            self.down1(e1),
            emb,
        )

        m = self.mid2(
            self.mid1(
                self.down2(e2),
                emb,
            ),
            emb,
        )

        d2 = self.dec2(
            torch.cat(
                [self.up2(m), e2],
                dim=1,
            ),
            emb,
        )

        d1 = self.dec1(
            torch.cat(
                [self.up1(d2), e1],
                dim=1,
            ),
            emb,
        )

        return self.out(d1) / s
