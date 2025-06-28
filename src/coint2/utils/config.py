"""Configuration utilities using Pydantic models."""

from pathlib import Path

import yaml  # type: ignore
from pydantic import BaseModel, DirectoryPath  # type: ignore


class PairSelectionConfig(BaseModel):
    """Configuration for pair selection parameters."""

    lookback_days: int
    coint_pvalue_threshold: float
    ssd_top_n: int


class BacktestConfig(BaseModel):
    """Configuration for backtesting parameters."""

    timeframe: str
    rolling_window: int
    zscore_threshold: float
    fill_limit_pct: float
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
