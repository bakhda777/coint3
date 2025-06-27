from pathlib import Path

from coint2.utils.config import AppConfig, load_config


def test_load_config():
    """Configuration file should load into AppConfig."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "main.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.pair_selection.lookback_days == 90
    assert cfg.backtest.rolling_window == 30

