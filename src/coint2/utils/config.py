"""Configuration utilities using Pydantic models."""

from pathlib import Path

import yaml  # type: ignore
from pydantic import BaseModel, DirectoryPath  # type: ignore


class PairSelectionConfig(BaseModel):
    """Configuration for pair selection parameters."""

    lookback_days: int
    coint_pvalue_threshold: float


class BacktestConfig(BaseModel):
    """Configuration for backtesting parameters."""

    timeframe: str
    rolling_window: int
    zscore_threshold: float
    fill_limit_pct: float


class AppConfig(BaseModel):
    """Top-level application configuration."""

    data_dir: DirectoryPath
    results_dir: Path
    pair_selection: PairSelectionConfig
    backtest: BacktestConfig


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


CONFIG = load_config(
    Path(__file__).resolve().parents[3] / "configs" / "main.yaml"
)
"""Singleton configuration loaded at import time."""