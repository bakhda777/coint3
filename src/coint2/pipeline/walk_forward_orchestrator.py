"""Walk-forward analysis orchestrator."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from coint2.core.data_loader import DataHandler
from coint2.engine.backtest_engine import PairBacktester
from coint2.pipeline.pair_scanner import find_cointegrated_pairs
from coint2.core import performance
from coint2.utils.config import CONFIG
from coint2.utils.logging_utils import get_logger


def run_walk_forward() -> Dict[str, float]:
    """Run walk-forward analysis and return aggregated performance metrics."""
    logger = get_logger("walk_forward")
    cfg = CONFIG

    handler = DataHandler(
        cfg.data_dir, cfg.backtest.timeframe, cfg.backtest.fill_limit_pct
    )

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
            cfg.pair_selection.coint_pvalue_threshold,
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
        for s1, s2, *_ in pairs:
            pair_data = handler.load_pair_data(s1, s2, testing_start, testing_end)
            bt = PairBacktester(
                pair_data,
                window=cfg.backtest.rolling_window,
                z_threshold=cfg.backtest.zscore_threshold,
            )
            bt.run()
            step_pnl = step_pnl.add(bt.get_results()["pnl"], fill_value=0)

        aggregated_pnl = pd.concat([aggregated_pnl, step_pnl])

        current_date = testing_start

    aggregated_pnl = aggregated_pnl.dropna()
    if aggregated_pnl.empty:
        return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_pnl": 0.0}

    cumulative = aggregated_pnl.cumsum()
    return {
        "sharpe_ratio": performance.sharpe_ratio(aggregated_pnl),
        "max_drawdown": performance.max_drawdown(cumulative),
        "total_pnl": cumulative.iloc[-1],
    }
