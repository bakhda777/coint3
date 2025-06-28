"""Mathematical helper utilities."""

from __future__ import annotations

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Calculate rolling beta of ``y`` relative to ``x``.

    The beta is computed as the rolling covariance between ``y`` and ``x``
    divided by the rolling variance of ``x``.

    Parameters
    ----------
    y : pd.Series
        Dependent variable series.
    x : pd.Series
        Independent variable series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Series of rolling beta values aligned to the right edge of the window.
    """
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling z-score for ``series``.

    The z-score is computed as the deviation of each value from the rolling
    mean divided by the rolling standard deviation.

    Parameters
    ----------
    series : pd.Series
        Time series to compute the z-score on.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Series of rolling z-score values aligned to the right edge of the
        window.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def calculate_ssd(normalized_prices: pd.DataFrame) -> pd.Series:
    """Compute pairwise sum of squared differences (SSD) between columns.

    Parameters
    ----------
    normalized_prices : pd.DataFrame
        DataFrame where each column is a normalized price series for a ticker.

    Returns
    -------
    pd.Series
        Series indexed by ``(symbol1, symbol2)`` with SSD values sorted in
        ascending order.
    """
    data = normalized_prices.to_numpy()
    columns = normalized_prices.columns

    dot_matrix = data.T @ data
    sum_sq = np.diag(dot_matrix)
    ssd_matrix = sum_sq[:, None] + sum_sq[None, :] - 2 * dot_matrix

    i_upper, j_upper = np.triu_indices_from(ssd_matrix, k=1)
    pairs = pd.MultiIndex.from_arrays([columns[i_upper], columns[j_upper]])
    ssd_values = ssd_matrix[i_upper, j_upper]
    return pd.Series(ssd_values, index=pairs).sort_values()
