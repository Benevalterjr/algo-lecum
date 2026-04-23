"""LECUM package."""

from .allocation import calculate_allocation, professional_allocation
from .analysis import determine_next_action, get_advanced_analysis, get_narrative_feedback
from .data import generate_market_candidates, get_market_context, pad_to_dim
from .models import Encoder, EnergyModel, LatentNarrator, Predictor
from .strategy import run_market_scanner, score_candidates
from .training import TrainConfig, build_models, generate_synthetic_data, set_seed, train_energy_epoch, train_jepa_epoch

__all__ = [
    "Encoder",
    "Predictor",
    "EnergyModel",
    "LatentNarrator",
    "TrainConfig",
    "build_models",
    "set_seed",
    "generate_synthetic_data",
    "train_jepa_epoch",
    "train_energy_epoch",
    "get_advanced_analysis",
    "determine_next_action",
    "get_narrative_feedback",
    "get_market_context",
    "pad_to_dim",
    "generate_market_candidates",
    "score_candidates",
    "run_market_scanner",
    "calculate_allocation",
    "professional_allocation",
]
