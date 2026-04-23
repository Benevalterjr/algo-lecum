"""Camada de dados de mercado com comportamento explícito para yfinance."""

from __future__ import annotations

import numpy as np
import torch
import yfinance as yf


def get_market_context(ticker: str, period: str = "1mo", interval: str = "1d") -> torch.Tensor | None:
    # auto_adjust explícito para evitar mudança silenciosa de default.
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        return None

    returns = data["Close"].pct_change().dropna()
    vol = returns.rolling(window=5).std().dropna()
    if len(returns) < 5 or len(vol) < 5:
        return None

    context = np.concatenate([returns.values[-5:], vol.values[-5:]])
    return torch.tensor(context, dtype=torch.float32)


def pad_to_dim(x: torch.Tensor, target_dim: int = 64) -> torch.Tensor:
    flat = x.flatten()
    if flat.numel() > target_dim:
        raise ValueError(f"Input has {flat.numel()} elements but target_dim={target_dim}")
    out = torch.zeros(target_dim, dtype=torch.float32)
    out[: flat.numel()] = flat
    return out


def generate_market_candidates(n: int = 5, dim: int = 64, scale: float = 0.05) -> list[torch.Tensor]:
    return [torch.randn(dim, dtype=torch.float32) * scale for _ in range(n)]
