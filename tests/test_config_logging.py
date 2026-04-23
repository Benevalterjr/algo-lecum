from lecum.config import default_config
from lecum.logging_utils import get_logger


def test_default_config_values():
    cfg = default_config()
    assert cfg.model.input_dim == 64
    assert cfg.train.lr == 1e-3
    assert cfg.market.lookback_returns == 5


def test_logger_has_handler():
    logger = get_logger("lecum.test")
    assert logger.handlers
