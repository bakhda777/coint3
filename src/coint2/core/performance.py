import pandas as pd


def sharpe_ratio(pnl: pd.Series, freq: int = 252) -> float:
    returns = pnl
    mean = returns.mean() * freq
    std = returns.std() * (freq ** 0.5)
    return mean / std if std != 0 else 0.0


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    cumulative = cumulative_pnl
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    return drawdown.min()
