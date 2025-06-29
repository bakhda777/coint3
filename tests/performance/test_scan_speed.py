import time
import pandas as pd
from pathlib import Path
from coint2.core.data_loader import _scan_parquet_files

def create_many_files(base: Path, n: int) -> None:
    idx = pd.date_range("2021-01-01", periods=1, freq="D")
    for i in range(n):
        part = base / f"symbol=AAA{i}" / "year=2021" / "month=01"
        part.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": idx, "close": [i]})
        df.to_parquet(part / f"data_{i}.parquet")


def test_scan_speed(tmp_path: Path) -> None:
    create_many_files(tmp_path, 10_000)
    start = time.perf_counter()
    ds = _scan_parquet_files(tmp_path)
    _ = ds.to_table()
    elapsed = time.perf_counter() - start
    assert elapsed < 30
