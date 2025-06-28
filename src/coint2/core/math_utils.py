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


def calculate_half_life(series: pd.Series) -> float:
    """Estimate the half-life of mean reversion for a time series.

    The half-life represents the time it takes for a deviation from the
    mean to reduce by half assuming an Ornstein-Uhlenbeck process. The
    implementation follows the common approach of regressing the first
    difference of the series on its lagged values.

    Parameters
    ----------
    series : pd.Series
        Input time series.

    Returns
    -------
    float
        Estimated half-life. ``np.inf`` is returned when the estimated
        mean reversion speed is non-negative.
    """
    # align lagged series with the differenced series
    y_lag = series.shift(1).dropna()
    delta_y = (series - y_lag).dropna()
    common_index = y_lag.index.intersection(delta_y.index)
    y_lag = y_lag.loc[common_index]
    delta_y = delta_y.loc[common_index]

    # add constant and run OLS regression using a lightweight
    # implementation to avoid external dependencies at runtime
    try:  # pragma: no cover - use statsmodels when available
        import statsmodels.api as sm  # type: ignore

        X = sm.add_constant(y_lag.to_numpy())
        model = sm.OLS(delta_y.to_numpy(), X)
        result = model.fit()
        lambda_coef = float(result.params[1])
    except Exception:  # fallback if statsmodels is unavailable
        X = np.column_stack([np.ones(len(y_lag)), y_lag.to_numpy()])
        beta, *_ = np.linalg.lstsq(X, delta_y.to_numpy(), rcond=None)
        lambda_coef = float(beta[1])

    if lambda_coef >= 0:
        return float(np.inf)

    return -np.log(2.0) / lambda_coef


def count_mean_crossings(series: pd.Series) -> int:
    """Count how many times a series crosses its mean value."""

    centered_series = series - series.mean()
    signs = np.sign(centered_series)
    # diff will be non-zero when sign changes
    return int(np.where(np.diff(signs) != 0)[0].size)
