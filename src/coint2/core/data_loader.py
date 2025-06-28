import dask.dataframe as dd
import pandas as pd  # type: ignore
from pathlib import Path
from typing import List

from coint2.utils.config import AppConfig


class DataHandler:
    """Utility class for loading local parquet price files."""

    def __init__(self, cfg: AppConfig) -> None:
        self.data_dir = Path(cfg.data_dir)
        self.fill_limit_pct = cfg.backtest.fill_limit_pct
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._all_data_cache: dd.DataFrame | None = None

    def clear_cache(self) -> None:
        """Clears the in-memory Dask DataFrame cache."""
        self._all_data_cache = None

    def get_all_symbols(self) -> List[str]:
        """Return list of symbols based on partition directory names."""
        if not self.data_dir.exists():
            return []

        symbols = []
        for path in self.data_dir.iterdir():
            if path.is_dir() and path.name.startswith("symbol="):
                symbols.append(path.name.replace("symbol=", ""))
        return sorted(symbols)

    def _load_full_dataset(self) -> dd.DataFrame:
        """Create a lazy Dask DataFrame from the partitioned dataset."""
        if self._all_data_cache is not None:
            return self._all_data_cache

        if not self.data_dir.exists():
            # empty Dask DataFrame
            self._all_data_cache = dd.from_pandas(pd.DataFrame(), npartitions=1)
            return self._all_data_cache

        ddf = dd.read_parquet(self.data_dir, engine="pyarrow")

        self._all_data_cache = ddf
        return ddf

    def load_all_data_for_period(self, lookback_days: int) -> pd.DataFrame:
        """Load close prices for all symbols for the given lookback period."""
        ddf = self._load_full_dataset()

        if not ddf.columns:
            return pd.DataFrame()

        end_date = ddf["timestamp"].max().compute()
        if pd.isna(end_date):
            return pd.DataFrame()

        start_date = end_date - pd.Timedelta(days=lookback_days)
        filtered_ddf = ddf[ddf["timestamp"] >= start_date]
        filtered_pdf = filtered_ddf.compute()
        if filtered_pdf.empty:
            return pd.DataFrame()

        filtered_pdf["timestamp"] = pd.to_datetime(filtered_pdf["timestamp"])
        wide = filtered_pdf.pivot_table(index="timestamp", columns="symbol", values="close")
        wide = wide.sort_index()
        return wide

    def load_pair_data(
        self,
        symbol1: str,
        symbol2: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Load and align data for two symbols within the given date range."""
        ddf = self._load_full_dataset()

        pair_ddf = ddf[ddf["symbol"].isin([symbol1, symbol2])]
        pair_pdf = pair_ddf.compute()

        if pair_pdf.empty:
            return pd.DataFrame()

        pair_pdf["timestamp"] = pd.to_datetime(pair_pdf["timestamp"])
        mask = (pair_pdf["timestamp"] >= pd.Timestamp(start_date)) & (
            pair_pdf["timestamp"] <= pd.Timestamp(end_date)
        )
        pair_pdf = pair_pdf.loc[mask]

        if pair_pdf.empty:
            return pd.DataFrame()

        wide_df = pair_pdf.pivot_table(index="timestamp", columns="symbol", values="close")

        if wide_df.empty:
            return pd.DataFrame()

        freq = pd.infer_freq(wide_df.index) or "D"
        wide_df = wide_df.asfreq(freq)
        limit = int(len(wide_df) * self.fill_limit_pct)
        wide_df = wide_df.ffill(limit=limit).bfill(limit=limit)

        return wide_df[[symbol1, symbol2]].dropna()

    def load_and_normalize_data(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Load data for all symbols in the range and normalize each column."""
        ddf = self._load_full_dataset()

        if not ddf.columns:
            return pd.DataFrame()

        pdf = ddf.compute()
        if pdf.empty:
            return pd.DataFrame()

        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        mask = (pdf["timestamp"] >= pd.Timestamp(start_date)) & (
            pdf["timestamp"] <= pd.Timestamp(end_date)
        )
        pdf = pdf.loc[mask]

        if pdf.empty:
            return pd.DataFrame()

        wide_df = pdf.pivot_table(index="timestamp", columns="symbol", values="close")
        wide_df = wide_df.sort_index()
        if wide_df.empty:
            return pd.DataFrame()

        def _normalize(series: pd.Series) -> pd.Series:
            max_val = series.max()
            min_val = series.min()
            if pd.isna(max_val) or pd.isna(min_val) or max_val == min_val:
                return pd.Series(0.0, index=series.index)
            return (series - min_val) / (max_val - min_val)

        normalized = wide_df.apply(_normalize, axis=0)
        return normalized
