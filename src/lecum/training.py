"""Treinamento e utilitários de preparação."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from .models import Encoder, Predictor, EnergyModel


@dataclass
class TrainConfig:
    input_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 128
    lr: float = 1e-3
    seed: int = 42


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(num_samples: int = 10_000, input_dim: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(num_samples, input_dim)
    y = x + 0.1 * torch.randn(num_samples, input_dim)
    return x, y


def build_models(config: TrainConfig) -> tuple[Encoder, Predictor, EnergyModel]:
    encoder = Encoder(config.input_dim, config.hidden_dim, config.latent_dim)
    predictor = Predictor(config.latent_dim)
    energy_model = EnergyModel(config.latent_dim, config.hidden_dim)
    return encoder, predictor, energy_model


def train_jepa_epoch(
    encoder: Encoder,
    predictor: Predictor,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module | None = None,
) -> float:
    loss_fn = loss_fn or nn.MSELoss()
    encoder.train()
    predictor.train()

    optimizer.zero_grad()
    z_x = encoder(x)
    z_y = encoder(y)
    z_pred = predictor(z_x)
    loss = loss_fn(z_pred, z_y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_energy_epoch(
    encoder: Encoder,
    energy_model: EnergyModel,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: optim.Optimizer,
    margin: float = 1.0,
) -> float:
    encoder.eval()
    energy_model.train()

    with torch.no_grad():
        z_x = encoder(x)
        z_y = encoder(y)

    optimizer.zero_grad()
    pos_energy = energy_model(z_x, z_y)
    shuffled = z_y[torch.randperm(z_y.size(0))]
    neg_energy = energy_model(z_x, shuffled)

    # loss margin contrastiva: positivos devem ter energia menor
    loss = torch.relu(pos_energy - neg_energy + margin).mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())
