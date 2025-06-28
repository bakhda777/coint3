import numpy as np
import pandas as pd
from scipy.stats import linregress

from coint2.core.math_utils import (
    rolling_beta,
    rolling_zscore,
    calculate_ssd,
    calculate_half_life,
    count_mean_crossings,
    half_life_numba,
    mean_crossings_numba,
)


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


def test_calculate_ssd():
    df = pd.DataFrame(
        {
            "A": [0, 0, 0],
            "B": [1, 1, 1],
            "C": [0, 2, 4],
        }
    )

    result = calculate_ssd(df)

    expected_index = pd.MultiIndex.from_tuples(
        [
            ("A", "B"),
            ("B", "C"),
            ("A", "C"),
        ]
    )
    expected = pd.Series([3, 11, 20], index=expected_index)

    pd.testing.assert_series_equal(result, expected)


def test_calculate_half_life_deterministic() -> None:
    phi = 0.8
    series = pd.Series([phi**i for i in range(10)])
    expected = -np.log(2) / (phi - 1)
    result = calculate_half_life(series)
    assert np.isclose(result, expected)


def test_count_mean_crossings() -> None:
    series = pd.Series([1, -1, 1, -1, 1, -1])
    assert count_mean_crossings(series) == 5


def test_half_life_numba():
    series = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    result = half_life_numba(series)
    assert np.isclose(result, 1.0)


def test_mean_crossings_numba():
    series = np.array([1, 2, 1, 0, -1, -2, -1, 0, 1])
    result = mean_crossings_numba(series)
    assert result == 2
