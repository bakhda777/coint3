import pandas as pd
from pathlib import Path
import pytest

from itertools import combinations

from coint2.core.data_loader import DataHandler
from coint2.pipeline.pair_scanner import find_cointegrated_pairs, _coint_test


def create_parquet_files(tmp_path: Path) -> None:
    idx = pd.date_range('2021-01-01', periods=20, freq='D')
    a = pd.Series(range(20), index=idx)
    b = a + 0.1  # cointegrated with A
    c = pd.Series(range(20, 0, -1), index=idx)

    for sym, series in [('A', a), ('B', b), ('C', c)]:
        part_dir = tmp_path / f'symbol={sym}' / 'year=2021' / 'month=01'
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'timestamp': idx, 'close': series})
        df.to_parquet(part_dir / 'data.parquet')


def test_find_cointegrated_pairs(tmp_path: Path) -> None:
    create_parquet_files(tmp_path)
    handler = DataHandler(tmp_path, '1d', fill_limit_pct=0.1)
    data = handler.load_all_data_for_period(lookback_days=20)

    # reference sequential implementation
    expected: dict[tuple[str, str], tuple[float, float, float]] = {}
    for s1, s2 in combinations(data.columns, 2):
        s1_series = data[s1].dropna()
        s2_series = data[s2].dropna()
        pval = _coint_test(s1_series, s2_series)
        if pval < 0.05:
            beta = s1_series.cov(s2_series) / s2_series.var()
            spread = s1_series - beta * s2_series
            expected[(s1, s2)] = (beta, spread.mean(), spread.std())

    start = data.index.min()
    end = data.index.max()
    pairs = find_cointegrated_pairs(handler, start, end, p_value_threshold=0.05)
    result = {(s1, s2): (beta, mean, std) for s1, s2, beta, mean, std in pairs}
    assert result.keys() == expected.keys()
    for pair in result:
        r_beta, r_mean, r_std = result[pair]
        e_beta, e_mean, e_std = expected[pair]
        assert r_beta == pytest.approx(e_beta)
        assert r_mean == pytest.approx(e_mean)
        assert r_std == pytest.approx(e_std)
