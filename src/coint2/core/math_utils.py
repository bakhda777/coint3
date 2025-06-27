"""Mathematical helper utilities."""

from __future__ import annotations

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