"""Command line interface for the coint2 package."""

from __future__ import annotations

import click

from pathlib import Path

from coint2.core.data_loader import DataHandler
from coint2.engine.backtest_engine import PairBacktester
from coint2.pipeline.walk_forward_orchestrator import run_walk_forward
from coint2.utils.config import load_config


@click.group()
def main() -> None:
    """Entry point for the ``coint2`` command."""


@main.command()
@click.option("--pair", required=True, help="Pair in the format SYMBOL1,SYMBOL2")
def backtest(pair: str) -> None:
    """Quick backtest over the entire dataset (for debugging only)."""

    cfg = load_config(Path("configs/main.yaml"))
    s1, s2 = [p.strip() for p in pair.split(",")]
    handler = DataHandler(cfg)

    ddf = handler._load_full_dataset()
    if not ddf.columns:
        click.echo("No data available")
        return

    end_date = ddf["timestamp"].max().compute()
    start_date = ddf["timestamp"].min().compute()

    data = handler.load_pair_data(s1, s2, start_date, end_date)
    if data.empty:
        click.echo("No data available for the pair")
        return

    beta = data[s1].cov(data[s2]) / data[s2].var()
    spread = data[s1] - beta * data[s2]
    mean = spread.mean()
    std = spread.std()

    bt = PairBacktester(
        data,
        beta=beta,
        spread_mean=mean,
        spread_std=std,
        z_threshold=cfg.backtest.zscore_threshold,
    )
    bt.run()
    metrics = bt.get_performance_metrics()
    for k, v in metrics.items():
        click.echo(f"{k}: {v}")


@main.command(name="run")
@click.option(
    "--config",
    "config_path",
    default="configs/main.yaml",
    help="Path to the configuration YAML file.",
    type=click.Path(exists=True),
)
def run_cmd(config_path: str) -> None:
    """Run walk-forward analysis pipeline."""

    cfg = load_config(Path(config_path))
    metrics = run_walk_forward(cfg)
    for key, value in metrics.items():
        click.echo(f"{key}: {value}")


if __name__ == "__main__":
    main()
