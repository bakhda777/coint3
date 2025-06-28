from pathlib import Path

from coint2.utils.config import AppConfig, load_config
from coint2.utils.config import BacktestConfig
from pydantic import ValidationError
import pytest


def test_load_config():
    """Configuration file should load into AppConfig."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "main.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.pair_selection.lookback_days == 90
    assert cfg.backtest.rolling_window == 30
    assert cfg.backtest.commission_pct == 0.001
    assert cfg.backtest.slippage_pct == 0.0005
    assert cfg.backtest.annualizing_factor == 365


def test_fill_limit_pct_validation() -> None:
    """fill_limit_pct should be between 0 and 1."""
    with pytest.raises(ValidationError):
        BacktestConfig(
            timeframe="1d",
            rolling_window=1,
            zscore_threshold=1.0,
            fill_limit_pct=1.5,
            commission_pct=0.001,
            slippage_pct=0.0005,
            annualizing_factor=365,
        )

