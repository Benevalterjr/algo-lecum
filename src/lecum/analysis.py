"""Funções de decisão e interpretação energética."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def get_advanced_analysis(energies: Iterable[float]) -> tuple[torch.Tensor, float, str]:
    energy_tensor = torch.tensor(list(energies), dtype=torch.float32)
    probs = F.softmax(-energy_tensor, dim=0)

    sorted_energies, _ = torch.sort(energy_tensor)
    gap = torch.abs(sorted_energies[0] - sorted_energies[1]).item()
    confidence_level = "HIGH" if gap > 0.1 else "LOW"

    return probs, gap, confidence_level


def determine_next_action(prob_val: float, gap_val: float) -> str:
    if prob_val > 0.6 and gap_val > 0.2:
        return "🚀 EXECUTE: Proceed with high-confidence implementation."
    if gap_val < 0.05:
        return "🔄 RE-SAMPLE: Generate more candidates to break the tie."
    if prob_val < 0.4:
        return "📚 EXPAND CONTEXT: Current latent features are insufficient for a clear path."
    return "⚖️ REVIEW: Human-in-the-loop validation recommended for top paths."


def get_narrative_feedback(energy_val: float) -> str:
    if energy_val < -0.2:
        return "✅ HIGHLY PLAUSIBLE: This scenario aligns with learned patterns."
    if energy_val < -0.1:
        return "⚠️ MODERATE: The scenario is plausible but noisy."
    return "❌ UNLIKELY: High energy detected for this state."
