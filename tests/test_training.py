import pytest

torch = pytest.importorskip("torch")

import torch.optim as optim

from lecum.training import (
    TrainConfig,
    build_models,
    generate_synthetic_data,
    set_seed,
    train_energy_epoch,
    train_jepa_epoch,
)


def test_training_steps_return_float_loss():
    set_seed(123)
    cfg = TrainConfig(input_dim=16, hidden_dim=32, latent_dim=32, lr=1e-3)
    encoder, predictor, energy_model = build_models(cfg)
    x, y = generate_synthetic_data(num_samples=128, input_dim=16)

    opt_jepa = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=cfg.lr)
    loss_jepa = train_jepa_epoch(encoder, predictor, x, y, opt_jepa)
    assert isinstance(loss_jepa, float)
    assert loss_jepa > 0

    opt_energy = optim.Adam(energy_model.parameters(), lr=cfg.lr)
    loss_energy = train_energy_epoch(encoder, energy_model, x, y, opt_energy)
    assert isinstance(loss_energy, float)
    assert torch.isfinite(torch.tensor(loss_energy))
