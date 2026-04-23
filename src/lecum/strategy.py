"""Scanner de ativos e seleção por energia."""

from __future__ import annotations

import pandas as pd
import torch

from .analysis import determine_next_action, get_advanced_analysis
from .data import get_market_context, generate_market_candidates, pad_to_dim
from .models import Encoder, EnergyModel


def score_candidates(
    encoder: Encoder,
    energy_model: EnergyModel,
    context: torch.Tensor,
    candidates: list[torch.Tensor],
) -> tuple[list[float], torch.Tensor, float, str, int]:
    with torch.no_grad():
        z_ctx = encoder(context.unsqueeze(0))
        energies: list[float] = []
        for c in candidates:
            z_c = encoder(c.unsqueeze(0))
            energies.append(float(energy_model(z_ctx, z_c).item()))

    probs, gap, conf = get_advanced_analysis(energies)
    best_idx = int(torch.argmax(probs).item())
    return energies, probs, gap, conf, best_idx


def run_market_scanner(
    tickers: list[str],
    encoder: Encoder,
    energy_model: EnergyModel,
    input_dim: int = 64,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for t in tickers:
        raw_ctx = get_market_context(t)
        if raw_ctx is None:
            continue

        context = pad_to_dim(raw_ctx, target_dim=input_dim)
        candidates = generate_market_candidates(n=5, dim=input_dim, scale=0.05)
        _, probs, gap, conf, best_idx = score_candidates(encoder, energy_model, context, candidates)

        best_prob = float(probs[best_idx].item())
        rows.append(
            {
                "Ticker": t,
                "Confidence_Prob": best_prob,
                "Energy_Gap": gap,
                "Status": conf,
                "Directive": determine_next_action(best_prob, gap),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Confidence_Prob", "Energy_Gap", "Status", "Directive"])

    return pd.DataFrame(rows).sort_values(by="Energy_Gap", ascending=False).reset_index(drop=True)
