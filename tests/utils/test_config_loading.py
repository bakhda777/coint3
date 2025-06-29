from pathlib import Path

import pytest
from pydantic import ValidationError

from coint2.utils.config import AppConfig, BacktestConfig, load_config


def test_load_config():
    """Configuration file should load into AppConfig."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "main.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.pair_selection.lookback_days == 90
    assert cfg.pair_selection.ssd_top_n == 10000
    assert cfg.pair_selection.min_half_life_days == 1
    assert cfg.pair_selection.max_half_life_days == 30
    assert cfg.pair_selection.min_mean_crossings == 12
    assert cfg.backtest.rolling_window == 30
    assert cfg.backtest.stop_loss_multiplier == 3.0
    assert cfg.backtest.commission_pct == 0.001
    assert cfg.backtest.slippage_pct == 0.0005
    assert cfg.backtest.annualizing_factor == 365
    assert cfg.portfolio.initial_capital == 10000.0
    assert cfg.portfolio.risk_per_trade_pct == 0.01
    assert cfg.portfolio.max_active_positions == 5


def test_fill_limit_pct_validation() -> None:
    """fill_limit_pct should be between 0 and 1."""
    with pytest.raises(ValidationError):
        BacktestConfig(
            timeframe="1d",
            rolling_window=1,
            zscore_threshold=1.0,
            stop_loss_multiplier=2.0,
            fill_limit_pct=1.5,
            commission_pct=0.001,
            slippage_pct=0.0005,
            annualizing_factor=365,
        )

