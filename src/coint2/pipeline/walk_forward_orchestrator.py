"""Walk-forward analysis orchestrator."""

from __future__ import annotations

import pandas as pd

from coint2.core import math_utils, performance
from coint2.core.data_loader import DataHandler
from coint2.engine.backtest_engine import PairBacktester
from coint2.utils.config import AppConfig
from coint2.utils.logging_utils import get_logger
from coint2.utils.visualization import (
    create_performance_report, 
    format_metrics_summary, 
    calculate_extended_metrics
)
from pathlib import Path


def run_walk_forward(cfg: AppConfig) -> dict[str, float]:
    """Run walk-forward analysis and return aggregated performance metrics."""
    logger = get_logger("walk_forward")

    handler = DataHandler(cfg)
    handler.clear_cache()

    start_date = pd.to_datetime(cfg.walk_forward.start_date)
    end_date = pd.to_datetime(cfg.walk_forward.end_date)

    full_range_start = start_date - pd.Timedelta(days=cfg.walk_forward.training_period_days)
    logger.info(f"Загрузка данных за период {full_range_start} - {end_date}")
    master_df = handler.preload_all_data(full_range_start, end_date)
    logger.info(f"Загружено данных: {master_df.shape}, символов: {len(master_df.columns)}")

    current_date = start_date
    aggregated_pnl = pd.Series(dtype=float)
    daily_pnl = []
    equity_data = []
    pair_count_data = []
    trade_stats = []  # Статистика по сделкам

    equity = cfg.portfolio.initial_capital
    equity_curve = [equity]
    
    # Добавляем начальную точку в equity_data
    equity_data.append((start_date, equity))

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
        logger.debug(f"Training slice для периода {training_start}-{training_end}: {training_slice.shape}")
        
        if training_slice.empty or len(training_slice.columns) < 2:
            logger.warning(f"Недостаточно данных для обучения: пустой slice или < 2 символов")
            pairs: list[tuple[str, str, float, float, float]] = []
        else:
            logger.debug(f"Нормализация данных для {len(training_slice.columns)} символов")
            normalized_training = (training_slice - training_slice.min()) / (
                training_slice.max() - training_slice.min()
            )
            # Удаляем столбцы с NaN (когда min == max, деление дает NaN)
            normalized_training = normalized_training.dropna(axis=1)
            logger.debug(f"После нормализации осталось {len(normalized_training.columns)} символов")
            
            if len(normalized_training.columns) < 2:
                logger.warning("После нормализации осталось менее 2 символов")
                pairs = []
            else:
                logger.debug(f"Расчет SSD для top-{cfg.pair_selection.ssd_top_n} пар")
                ssd = math_utils.calculate_ssd(
                    normalized_training, top_k=cfg.pair_selection.ssd_top_n
                )
                logger.debug(f"SSD рассчитан для {len(ssd)} пар")
                
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
                
                logger.debug(f"Сформировано {len(pairs)} пар после базовых проверок")

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

        step_pnl = pd.Series(dtype=float)
        total_step_pnl = 0.0

        if active_pairs:
            capital_per_pair = equity * cfg.portfolio.risk_per_position_pct
        else:
            capital_per_pair = 0.0

        # Собираем данные для отчета
        period_label = f"{training_start.strftime('%m/%d')}-{testing_end.strftime('%m/%d')}"
        pair_count_data.append((period_label, len(active_pairs)))

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
            results = bt.get_results()
            pnl_series = results["pnl"] * capital_per_pair
            step_pnl = step_pnl.add(pnl_series, fill_value=0)
            total_step_pnl += pnl_series.sum()
            
            # Собираем статистику по сделкам для данной пары
            if isinstance(results, dict):
                trades = results.get("trades", pd.Series())
                positions = results.get("position", pd.Series())
                costs = results.get("costs", pd.Series())
            else:
                # Если results - DataFrame, используем колонки
                trades = results.get("trades", results.get("trades", pd.Series()))
                positions = results.get("position", results.get("position", pd.Series()))
                costs = results.get("costs", results.get("costs", pd.Series()))
            
            # Считаем количество открытий/закрытий позиций
            if not positions.empty:
                position_changes = positions.diff().fillna(0).abs()
                trade_opens = (position_changes > 0).sum()
            else:
                trade_opens = 0
            
            # Всегда добавляем статистику по парам, даже если сделок не было
            pair_pnl = pnl_series.sum()
            pair_costs = costs.sum() if not costs.empty else 0
            
            trade_stats.append({
                'pair': f'{s1}-{s2}',
                'period': period_label,
                'total_pnl': pair_pnl,
                'total_costs': pair_costs,
                'trade_count': trade_opens,
                'avg_pnl_per_trade': pair_pnl / max(trade_opens, 1),
                'win_days': (pnl_series > 0).sum(),
                'lose_days': (pnl_series < 0).sum(),
                'total_days': len(pnl_series),
                'max_daily_gain': pnl_series.max(),
                'max_daily_loss': pnl_series.min()
            })

        # Сохраняем дневной P&L для периода
        if not step_pnl.empty:
            running_equity = equity
            for date, pnl in step_pnl.items():
                daily_pnl.append((date, pnl))
                running_equity += pnl
                equity_data.append((date, running_equity))

        aggregated_pnl = pd.concat([aggregated_pnl, step_pnl])
        equity += total_step_pnl
        equity_curve.append(equity)

        current_date = testing_end

    aggregated_pnl = aggregated_pnl.dropna()
    
    # Создаем серии для анализа
    if daily_pnl:
        dates, pnls = zip(*daily_pnl)
        pnl_series = pd.Series(pnls, index=pd.to_datetime(dates))
        pnl_series = pnl_series.groupby(pnl_series.index.date).sum()  # Агрегируем по дням
        pnl_series.index = pd.to_datetime(pnl_series.index)
    else:
        pnl_series = pd.Series(dtype=float)
    
    if equity_data:
        eq_dates, eq_values = zip(*equity_data)
        equity_series = pd.Series(eq_values, index=pd.to_datetime(eq_dates))
    else:
        equity_series = pd.Series([cfg.portfolio.initial_capital])
    
    # Базовые метрики
    if aggregated_pnl.empty:
        base_metrics = {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_pnl": 0.0}
    else:
        cumulative = aggregated_pnl.cumsum()
        base_metrics = {
            "sharpe_ratio": performance.sharpe_ratio(
                aggregated_pnl, cfg.backtest.annualizing_factor
            ),
            "max_drawdown": performance.max_drawdown(cumulative),
            "total_pnl": cumulative.iloc[-1] if not cumulative.empty else 0.0,
        }
    
    # Расширенные метрики
    extended_metrics = calculate_extended_metrics(pnl_series, equity_series)
    
    # Статистика по сделкам
    trade_metrics = {}
    if trade_stats:
        trades_df = pd.DataFrame(trade_stats)
        trade_metrics = {
            'total_trades': trades_df['trade_count'].sum(),
            'total_pairs_traded': len(trades_df['pair'].unique()),
            'total_costs': trades_df['total_costs'].sum(),
            'avg_trades_per_pair': trades_df['trade_count'].mean(),
            'win_rate_trades': trades_df['win_days'].sum() / max(trades_df['total_days'].sum(), 1),
            'best_pair_pnl': trades_df['total_pnl'].max(),
            'worst_pair_pnl': trades_df['total_pnl'].min(),
            'avg_pnl_per_pair': trades_df['total_pnl'].mean(),
        }
    
    all_metrics = {**base_metrics, **extended_metrics, **trade_metrics}
    
    # Создаем отчеты
    results_dir = Path(cfg.results_dir)
    
    try:
        # Создаем визуальный отчет
        create_performance_report(
            equity_curve=equity_series,
            pnl_series=pnl_series,
            metrics=all_metrics,
            pair_counts=pair_count_data,
            results_dir=results_dir,
            strategy_name="CointegrationStrategy"
        )
        
        # Выводим красивые итоги
        summary = format_metrics_summary(all_metrics)
        print(summary)
        
        # Сохраняем метрики в CSV
        metrics_df = pd.DataFrame([all_metrics])
        metrics_df.to_csv(results_dir / "strategy_metrics.csv", index=False)
        print(f"📋 Метрики сохранены: {results_dir / 'strategy_metrics.csv'}")
        
        # Сохраняем временные ряды
        if not pnl_series.empty:
            pnl_series.to_csv(results_dir / "daily_pnl.csv", header=['PnL'])
            print(f"📈 Дневные P&L сохранены: {results_dir / 'daily_pnl.csv'}")
        
        if not equity_series.empty:
            equity_series.to_csv(results_dir / "equity_curve.csv", header=['Equity'])
            print(f"💰 Кривая капитала сохранена: {results_dir / 'equity_curve.csv'}")
        
        # Сохраняем статистику по сделкам
        if trade_stats:
            trades_df = pd.DataFrame(trade_stats)
            trades_df.to_csv(results_dir / "trade_statistics.csv", index=False)
            print(f"🔄 Статистика по сделкам сохранена: {results_dir / 'trade_statistics.csv'}")
            
    except Exception as e:
        logger.error(f"Ошибка при создании отчетов: {e}")
    
    return base_metrics
