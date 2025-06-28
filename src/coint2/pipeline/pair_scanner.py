from typing import List, Tuple

import pandas as pd
import dask
from dask import delayed
from statsmodels.tsa.stattools import coint

from coint2.core import math_utils
from coint2.utils.config import AppConfig


def _coint_test(series1: pd.Series, series2: pd.Series) -> float:
    """Run cointegration test and return p-value."""
    _score, pvalue, _ = coint(series1, series2)
    return pvalue


@delayed
def _test_pair_for_coint(
    handler,
    symbol1: str,
    symbol2: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    p_value_threshold: float,
) -> Tuple[str, str, float, float, float] | None:
    """Lazy test for a single pair using provided handler and dates."""
    pair_data = handler.load_pair_data(symbol1, symbol2, start_date, end_date)
    if pair_data.empty or len(pair_data.columns) < 2:
        return None

    pvalue = _coint_test(pair_data[symbol1].dropna(), pair_data[symbol2].dropna())
    if pvalue >= p_value_threshold:
        return None

    y = pair_data[symbol1]
    x = pair_data[symbol2]
    beta = y.cov(x) / x.var()
    spread = y - beta * x
    mean = spread.mean()
    std = spread.std()
    return symbol1, symbol2, beta, mean, std


def find_cointegrated_pairs(
    handler,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cfg: AppConfig,
) -> List[Tuple[str, str, float, float, float]]:
    """Find cointegrated pairs using SSD pre-filtering."""

    p_value_threshold = cfg.pair_selection.coint_pvalue_threshold

    normalized = handler.load_and_normalize_data(start_date, end_date)
    if normalized.empty or len(normalized.columns) < 2:
        return []

    ssd = math_utils.calculate_ssd(normalized)
    top_pairs = ssd.head(cfg.pair_selection.ssd_top_n).index.tolist()

    lazy_results = []
    for s1, s2 in top_pairs:
        task = _test_pair_for_coint(
            handler,
            s1,
            s2,
            start_date,
            end_date,
            p_value_threshold,
        )
        lazy_results.append(task)

    results = dask.compute(*lazy_results, scheduler="processes")
    return [r for r in results if r is not None]
