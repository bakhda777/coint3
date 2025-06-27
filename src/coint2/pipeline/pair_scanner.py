from itertools import combinations
from typing import List, Tuple

import pandas as pd  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from statsmodels.tsa.stattools import coint  # type: ignore


def _coint_test(series1: pd.Series, series2: pd.Series) -> float:
    """Run cointegration test and return p-value."""
    _score, pvalue, _ = coint(series1, series2)
    return pvalue


def find_cointegrated_pairs(data: pd.DataFrame, p_value_threshold: float) -> List[Tuple[str, str]]:
    """Find cointegrated pairs among columns of the provided data."""
    symbols = list(data.columns)
    pairs = list(combinations(symbols, 2))

    def check_pair(pair: Tuple[str, str]):
        s1, s2 = pair
        pvalue = _coint_test(data[s1].dropna(), data[s2].dropna())
        return pair if pvalue < p_value_threshold else None

    results = Parallel(n_jobs=-1)(delayed(check_pair)(p) for p in pairs)
    return [r for r in results if r is not None]