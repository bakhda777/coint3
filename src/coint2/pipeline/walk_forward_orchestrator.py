"""Walk-forward analysis orchestrator."""

from __future__ import annotations

import pandas as pd

from coint2.core import math_utils, performance
from coint2.core.data_loader import DataHandler
from coint2.engine.backtest_engine import PairBacktester
from coint2.utils.config import AppConfig
from coint2.utils.logging_utils import get_logger


def run_walk_forward(cfg: AppConfig) -> dict[str, float]:
    """Run walk-forward analysis and return aggregated performance metrics."""
    logger = get_logger("walk_forward")

    handler = DataHandler(cfg)
    handler.clear_cache()

    start_date = pd.to_datetime(cfg.walk_forward.start_date)
    end_date = pd.to_datetime(cfg.walk_forward.end_date)

    full_range_start = start_date - pd.Timedelta(days=cfg.walk_forward.training_period_days)
    master_df = handler.preload_all_data(full_range_start, end_date)

    current_date = start_date
    aggregated_pnl = pd.Series(dtype=float)

    equity = cfg.portfolio.initial_capital
    equity_curve = [equity]

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

        training_slice = master_df.loc[training_start:training_end]
        if training_slice.empty or len(training_slice.columns) < 2:
            pairs: list[tuple[str, str, float, float, float]] = []
        else:
            normalized_training = (training_slice - training_slice.min()) / (
                training_slice.max() - training_slice.min()
            )
            ssd = math_utils.calculate_ssd(
                normalized_training, top_k=cfg.pair_selection.ssd_top_n
            )
            pairs = []
            for s1, s2 in ssd.index:
                pair_train = training_slice[[s1, s2]].dropna()
                if pair_train.empty or pair_train[s2].var() == 0:
                    continue
                beta = pair_train[s1].cov(pair_train[s2]) / pair_train[s2].var()
                spread = pair_train[s1] - beta * pair_train[s2]
                mean = spread.mean()
                std = spread.std()
                pairs.append((s1, s2, beta, mean, std))

        logger.info(
            "Walk-forward step train %s-%s, test %s-%s, %d pairs",
            training_start.date(),
            training_end.date(),
            testing_start.date(),
            testing_end.date(),
            len(pairs),
        )

        sorted_pairs = sorted(pairs)
        active_pairs = sorted_pairs[: cfg.portfolio.max_active_positions]

        total_risk_capital = equity * cfg.portfolio.risk_per_trade_pct

        step_pnl = pd.Series(dtype=float)
        total_step_pnl = 0.0

        if active_pairs:
            capital_per_pair = total_risk_capital / len(active_pairs)
        else:
            capital_per_pair = 0.0

        for s1, s2, beta, mean, std in active_pairs:
            pair_data = master_df.loc[testing_start:testing_end, [s1, s2]].dropna()
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
            pnl_series = bt.get_results()["pnl"] * capital_per_pair
            step_pnl = step_pnl.add(pnl_series, fill_value=0)
            total_step_pnl += pnl_series.sum()

        aggregated_pnl = pd.concat([aggregated_pnl, step_pnl])
        equity += total_step_pnl
        equity_curve.append(equity)

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
