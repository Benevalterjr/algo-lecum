"""Modelos centrais do protótipo LECUM."""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        return self.net(z_context)


class EnergyModel(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_context: torch.Tensor, z_candidate: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_context, z_candidate], dim=-1)
        return self.net(x)


class LatentNarrator(nn.Module):
    def __init__(self, latent_dim: int = 128, vocab_size: int = 50):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, vocab_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
