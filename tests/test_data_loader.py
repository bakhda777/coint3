import pandas as pd
from pathlib import Path

from coint2.core.data_loader import DataHandler


def create_parquet_files(tmp_path: Path):
    idx = pd.date_range('2021-01-01', periods=10, freq='D')
    a = pd.Series(range(10), index=idx)
    b = a + 0.5  # cointegrated with a
    c = pd.Series(range(0, 20, 2), index=idx)
    for sym, series in [('A', a), ('B', b), ('C', c)]:
        df = pd.DataFrame({'timestamp': idx, 'close': series})
        (tmp_path / '1d').mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp_path / '1d' / f'{sym}.parquet')


def test_data_handler(tmp_path: Path):
    create_parquet_files(tmp_path)
    handler = DataHandler(tmp_path, '1d', fill_limit_pct=0.1)
    symbols = handler.get_all_symbols()
    assert set(symbols) == {'A', 'B', 'C'}

    df_all = handler.load_all_data_for_period(lookback_days=5)
    assert set(df_all.columns) == {'A', 'B', 'C'}
    assert len(df_all) == 5

    df_pair = handler.load_pair_data('A', 'B')
    assert {'A', 'B'} == set(df_pair.columns)

