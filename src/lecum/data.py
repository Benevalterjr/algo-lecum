"""Camada de dados de mercado com comportamento explícito para yfinance."""

from __future__ import annotations

import numpy as np
import torch
import yfinance as yf

from .config import MarketConfig


def get_market_context(ticker: str, config: MarketConfig | None = None) -> torch.Tensor | None:
    cfg = config or MarketConfig()
    data = yf.download(
        ticker,
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        return None

    returns = data["Close"].pct_change().dropna()
    vol = returns.rolling(window=cfg.lookback_volatility).std().dropna()
    if len(returns) < cfg.lookback_returns or len(vol) < cfg.lookback_volatility:
        return None

    context = np.concatenate(
        [
            returns.values[-cfg.lookback_returns :],
            vol.values[-cfg.lookback_volatility :],
        ]
    )
    return torch.tensor(context, dtype=torch.float32)


def pad_to_dim(x: torch.Tensor, target_dim: int = 64) -> torch.Tensor:
    flat = x.flatten()
    if flat.numel() > target_dim:
        raise ValueError(f"Input has {flat.numel()} elements but target_dim={target_dim}")
    out = torch.zeros(target_dim, dtype=torch.float32)
    out[: flat.numel()] = flat
    return out


def generate_market_candidates(
    n: int = 5,
    dim: int = 64,
    scale: float = 0.05,
) -> list[torch.Tensor]:
    return [torch.randn(dim, dtype=torch.float32) * scale for _ in range(n)]
