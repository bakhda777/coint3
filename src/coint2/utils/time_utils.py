import pandas as pd


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a timezone naive ``DatetimeIndex``.

    The index is converted to ``DatetimeIndex`` if needed, timezone
    information is stripped and the index is sorted.
    """
    result = df.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)
    result = result.sort_index()
    return result


def infer_frequency(idx: pd.Index) -> str | None:
    """Infer frequency of a ``DatetimeIndex`` safely."""
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    try:
        return pd.infer_freq(idx)
    except (TypeError, ValueError):
        return None
