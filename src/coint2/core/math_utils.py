import pandas as pd
import numpy as np


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Compute rolling beta of y ~ x using OLS."""
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov / var
    return beta


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std
