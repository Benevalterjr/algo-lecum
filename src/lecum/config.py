"""Configuração centralizada do projeto LECUM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 128


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    margin: float = 1.0
    seed: int = 42
    jepa_epochs: int = 10
    energy_epochs: int = 10


@dataclass(frozen=True)
class MarketConfig:
    period: str = "1mo"
    interval: str = "1d"
    lookback_returns: int = 5
    lookback_volatility: int = 5
    candidate_count: int = 5
    candidate_scale: float = 0.05


@dataclass(frozen=True)
class AllocationConfig:
    max_exposure: float = 0.85
    temperature: float = 1.5


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    market: MarketConfig = MarketConfig()
    allocation: AllocationConfig = AllocationConfig()


def default_config() -> AppConfig:
    return AppConfig()
