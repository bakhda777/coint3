"""Configuration utilities using Pydantic models."""

from pathlib import Path

import yaml  # type: ignore
from pydantic import BaseModel, DirectoryPath, Field  # type: ignore


class PairSelectionConfig(BaseModel):
    """Configuration for pair selection parameters."""

    lookback_days: int
    coint_pvalue_threshold: float
    ssd_top_n: int
    min_half_life_days: int
    max_half_life_days: int
    min_mean_crossings: int


class PortfolioConfig(BaseModel):
    """Configuration for portfolio and risk management."""

    initial_capital: float
    risk_per_trade_pct: float
    max_active_positions: int


class BacktestConfig(BaseModel):
    """Configuration for backtesting parameters."""

    timeframe: str
    rolling_window: int
    zscore_threshold: float
    stop_loss_multiplier: float
    fill_limit_pct: float = Field(..., gt=0.0, lt=1.0)
    commission_pct: float  # Новое поле
    slippage_pct: float  # Новое поле
    annualizing_factor: int  # Новое поле


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward analysis."""

    start_date: str
    end_date: str
    training_period_days: int
    testing_period_days: int


class AppConfig(BaseModel):
    """Top-level application configuration."""

    data_dir: DirectoryPath
    results_dir: Path
    portfolio: PortfolioConfig
    pair_selection: PairSelectionConfig
    backtest: BacktestConfig
    walk_forward: WalkForwardConfig


def load_config(path: Path) -> AppConfig:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    AppConfig
        Parsed configuration object.
    """
    with path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    return AppConfig(**raw_cfg)
