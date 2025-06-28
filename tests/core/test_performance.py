import numpy as np
import pandas as pd

from coint2.core import performance


def test_sharpe_ratio_uses_annualizing_factor():
    pnl = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    daily_sharpe = pnl.mean() / pnl.std(ddof=0)

    result_252 = performance.sharpe_ratio(pnl, 252)
    result_365 = performance.sharpe_ratio(pnl, 365)

    assert np.isclose(result_252, daily_sharpe * np.sqrt(252))
    assert np.isclose(result_365, daily_sharpe * np.sqrt(365))
