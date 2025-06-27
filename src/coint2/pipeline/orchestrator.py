"""Pipeline orchestrator tying together scanning and backtesting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from coint2.core.data_loader import DataHandler
from coint2.engine.backtest_engine import PairBacktester
from coint2.pipeline.pair_scanner import find_cointegrated_pairs
from coint2.utils.config import CONFIG
from coint2.utils.logging_utils import get_logger


def run_full_pipeline() -> List[Dict[str, object]]:
    """Execute scanning and backtesting for all detected pairs."""

    logger = get_logger("orchestrator")
    cfg = CONFIG

    handler = DataHandler(
        cfg.data_dir, cfg.backtest.timeframe, cfg.backtest.fill_limit_pct
    )

    logger.info("Loading data for %s days", cfg.pair_selection.lookback_days)
    data = handler.load_all_data_for_period(cfg.pair_selection.lookback_days)
    if data.empty:
        logger.warning("No data available to scan")
        return []

    logger.info("Scanning for cointegrated pairs")
    pairs = find_cointegrated_pairs(
        data, cfg.pair_selection.coint_pvalue_threshold
    )
    logger.info("Found %d pairs", len(pairs))

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, object]] = []
    for s1, s2 in pairs:
        logger.info("Backtesting %s-%s", s1, s2)
        pair_data = handler.load_pair_data(s1, s2)
        bt = PairBacktester(
            pair_data,
            window=cfg.backtest.rolling_window,
            z_threshold=cfg.backtest.zscore_threshold,
        )
        bt.run()
        metrics = bt.get_performance_metrics()
        logger.info("Metrics for %s-%s: %s", s1, s2, metrics)

        metrics_path = results_dir / f"{s1}_{s2}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f)

        all_metrics.append({"pair": (s1, s2), **metrics})

    return all_metrics

