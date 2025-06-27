from itertools import combinations
from typing import List, Tuple

import pandas as pd
import dask
from dask import delayed
from statsmodels.tsa.stattools import coint


def _coint_test(series1: pd.Series, series2: pd.Series) -> float:
    """Run cointegration test and return p-value."""
    _score, pvalue, _ = coint(series1, series2)
    return pvalue


@delayed
def _test_pair_for_coint(pair_data: pd.DataFrame, p_value_threshold: float) -> Tuple[str, str] | None:
    """Lazy test for a single pair using already loaded price data."""
    if pair_data.empty or len(pair_data.columns) < 2:
        return None

    s1, s2 = pair_data.columns
    pvalue = _coint_test(pair_data[s1].dropna(), pair_data[s2].dropna())
    return (s1, s2) if pvalue < p_value_threshold else None


def find_cointegrated_pairs(handler, lookback_days: int, p_value_threshold: float) -> List[Tuple[str, str]]:
    """Generate and compute dask tasks to find cointegrated pairs."""
    all_symbols = handler.get_all_symbols()
    all_pairs = list(combinations(all_symbols, 2))

    lazy_results = []
    for s1, s2 in all_pairs:
        pair_df = handler.load_pair_data(s1, s2)
        if not pair_df.empty:
            end = pair_df.index.max()
            start = end - pd.Timedelta(days=lookback_days)
            pair_df = pair_df[pair_df.index >= start]

        task = _test_pair_for_coint(pair_df, p_value_threshold)
        lazy_results.append(task)

    results = dask.compute(*lazy_results, scheduler="processes")
    return [r for r in results if r is not None]
