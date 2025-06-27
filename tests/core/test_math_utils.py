import numpy as np
import pandas as pd
from scipy.stats import linregress

from coint2.core.math_utils import rolling_beta, rolling_zscore


def test_rolling_beta_matches_linregress():
    x = pd.Series(np.arange(10, dtype=float))
    y = x * 2 + 1  # perfect linear relation with beta=2
    window = 5
    beta = rolling_beta(y, x, window)

    # expected beta computed via linregress on each window
    expected = []
    for i in range(len(x)):
        if i + 1 < window:
            expected.append(np.nan)
        else:
            sl, _, _, _, _ = linregress(x[i - window + 1 : i + 1], y[i - window + 1 : i + 1])
            expected.append(sl)
    expected = pd.Series(expected, index=x.index)

    pd.testing.assert_series_equal(beta, expected)


def test_rolling_zscore_basic():
    series = pd.Series([1, 2, 3, 4, 5, 6])
    z = rolling_zscore(series, 3)
    means = series.rolling(3).mean()
    stds = series.rolling(3).std()
    expected = (series - means) / stds
    pd.testing.assert_series_equal(z, expected)