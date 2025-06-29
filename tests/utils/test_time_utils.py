import pandas as pd
import pytest

from coint2.utils.time_utils import ensure_datetime_index, infer_frequency


def test_ensure_datetime_index_sorts_and_drops_tz() -> None:
    idx = pd.date_range("2021-01-01", periods=2, freq="D", tz="UTC")[::-1]
    df = pd.DataFrame({"a": [1, 2]}, index=idx)

    result = ensure_datetime_index(df)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is None
    expected = list(pd.date_range("2021-01-01", periods=2, freq="D"))
    assert list(result.index) == expected


@pytest.mark.parametrize("freq", ["D", "H", "15T"])
def test_infer_frequency_regular(freq: str) -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq=freq)
    assert infer_frequency(idx) == freq


def test_infer_frequency_irregular() -> None:
    idx = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-04"])
    assert infer_frequency(idx) is None

