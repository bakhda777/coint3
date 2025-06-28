"""Walk-forward analysis orchestrator."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from coint2.core.data_loader import DataHandler
from coint2.engine.backtest_engine import PairBacktester
from coint2.pipeline.pair_scanner import find_cointegrated_pairs
from coint2.core import performance
from coint2.utils.config import AppConfig
from coint2.utils.logging_utils import get_logger


def run_walk_forward(cfg: AppConfig) -> Dict[str, float]:
    """Run walk-forward analysis and return aggregated performance metrics."""
    logger = get_logger("walk_forward")

    handler = DataHandler(cfg)
    handler.clear_cache()

    start_date = pd.to_datetime(cfg.walk_forward.start_date)
    end_date = pd.to_datetime(cfg.walk_forward.end_date)

    current_date = start_date
    aggregated_pnl = pd.Series(dtype=float)

    while current_date < end_date:
        training_start = current_date
        training_end = training_start + pd.Timedelta(
            days=cfg.walk_forward.training_period_days
        )

        testing_start = training_end
        testing_end = testing_start + pd.Timedelta(
            days=cfg.walk_forward.testing_period_days
        )

        if testing_end > end_date:
            break

        pairs = find_cointegrated_pairs(
            handler,
            training_start,
            training_end,
            cfg,
        )

        logger.info(
            "Walk-forward step train %s-%s, test %s-%s, %d pairs",
            training_start.date(),
            training_end.date(),
            testing_start.date(),
            testing_end.date(),
            len(pairs),
        )

        step_pnl = pd.Series(dtype=float)
        for s1, s2, beta, mean, std in pairs:
            pair_data = handler.load_pair_data(s1, s2, testing_start, testing_end)
            bt = PairBacktester(
                pair_data,
                beta=beta,
                spread_mean=mean,
                spread_std=std,
                z_threshold=cfg.backtest.zscore_threshold,
                commission_pct=cfg.backtest.commission_pct,
                slippage_pct=cfg.backtest.slippage_pct,
                annualizing_factor=cfg.backtest.annualizing_factor,
            )
            bt.run()
            step_pnl = step_pnl.add(bt.get_results()["pnl"], fill_value=0)

        aggregated_pnl = pd.concat([aggregated_pnl, step_pnl])

        current_date = testing_end

    aggregated_pnl = aggregated_pnl.dropna()
    if aggregated_pnl.empty:
        return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_pnl": 0.0}

    cumulative = aggregated_pnl.cumsum()
    return {
        "sharpe_ratio": performance.sharpe_ratio(
            aggregated_pnl, cfg.backtest.annualizing_factor
        ),
        "max_drawdown": performance.max_drawdown(cumulative),
        "total_pnl": cumulative.iloc[-1],
    }
