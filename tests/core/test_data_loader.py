import pandas as pd
from pathlib import Path

from coint2.core.data_loader import DataHandler


def create_dataset(tmp_path: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    for sym, shift in [("AAA", 0), ("BBB", 1)]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        series = pd.Series(range(5), index=idx) + shift
        df = pd.DataFrame({"timestamp": idx, "close": series})
        df.to_parquet(part_dir / "data.parquet")


def test_load_all_data_for_period(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    handler = DataHandler(tmp_path, "1d", fill_limit_pct=0.1)

    result = handler.load_all_data_for_period(lookback_days=2)

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf = pdf.sort_values("timestamp")
    end_date = pdf["timestamp"].max()
    start_date = end_date - pd.Timedelta(days=2)
    filtered = pdf[pdf["timestamp"] >= start_date]
    expected = filtered.pivot_table(index="timestamp", columns="symbol", values="close")
    expected = expected.sort_index()

    pd.testing.assert_frame_equal(result, expected)


def test_load_pair_data(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    handler = DataHandler(tmp_path, "1d", fill_limit_pct=0.1)

    result = handler.load_pair_data(
        "AAA",
        "BBB",
        pd.Timestamp("2021-01-02"),
        pd.Timestamp("2021-01-04"),
    )

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf = pdf[pdf["symbol"].isin(["AAA", "BBB"])]
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    expected = pdf.pivot_table(index="timestamp", columns="symbol", values="close")
    expected = expected.asfreq(pd.infer_freq(expected.index) or "D")
    limit = int(len(expected) * 0.1)
    expected = expected.ffill(limit=limit).bfill(limit=limit)
    expected = expected[["AAA", "BBB"]].dropna()
    expected = expected.loc[pd.Timestamp("2021-01-02"): pd.Timestamp("2021-01-04")]

    pd.testing.assert_frame_equal(result, expected)

