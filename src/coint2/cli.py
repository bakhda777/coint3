"""Command line interface for the coint2 package."""

from __future__ import annotations

import click
import pandas as pd

from coint2.core.data_loader import DataHandler
from coint2.engine.backtest_engine import PairBacktester
from coint2.pipeline.orchestrator import run_full_pipeline
from coint2.pipeline.pair_scanner import find_cointegrated_pairs
from coint2.utils.config import CONFIG
from coint2.utils.logging_utils import get_logger


@click.group()
def main() -> None:
    """Entry point for the ``coint2`` command."""


@main.command()
def scan() -> None:
    """Scan available symbols for cointegrated pairs."""

    logger = get_logger("scan")
    cfg = CONFIG
    handler = DataHandler(cfg.data_dir, cfg.backtest.timeframe, cfg.backtest.fill_limit_pct)

    ddf = handler._load_full_dataset()
    if not ddf.columns:
        click.echo("No data available")
        return

    end_date = ddf["timestamp"].max().compute()
    start_date = end_date - pd.Timedelta(days=cfg.pair_selection.lookback_days)

    pairs = find_cointegrated_pairs(
        handler,
        start_date,
        end_date,
        cfg.pair_selection.coint_pvalue_threshold,
    )
    if not pairs:
        click.echo("No cointegrated pairs found")
        return
    logger.info("Found %d cointegrated pairs", len(pairs))
    for s1, s2 in pairs:
        click.echo(f"{s1},{s2}")


@main.command()
@click.option("--pair", required=True, help="Pair in the format SYMBOL1,SYMBOL2")
def backtest(pair: str) -> None:
    """Backtest a single pair."""

    cfg = CONFIG
    s1, s2 = [p.strip() for p in pair.split(",")]
    handler = DataHandler(
        cfg.data_dir, cfg.backtest.timeframe, cfg.backtest.fill_limit_pct
    )

    ddf = handler._load_full_dataset()
    if not ddf.columns:
        click.echo("No data available")
        return

    end_date = ddf["timestamp"].max().compute()
    start_date = ddf["timestamp"].min().compute()

    data = handler.load_pair_data(s1, s2, start_date, end_date)
    bt = PairBacktester(
        data,
        window=cfg.backtest.rolling_window,
        z_threshold=cfg.backtest.zscore_threshold,
    )
    bt.run()
    metrics = bt.get_performance_metrics()
    for k, v in metrics.items():
        click.echo(f"{k}: {v}")


@main.command(name="run-pipeline")
def run_pipeline_cmd() -> None:
    """Run full pipeline of scanning and backtesting."""

    results = run_full_pipeline()
    for entry in results:
        pair = entry.get("pair")
        if pair:
            click.echo(f"{pair[0]},{pair[1]}")
        for key, value in entry.items():
            if key != "pair":
                click.echo(f"  {key}: {value}")


if __name__ == "__main__":
    main()
