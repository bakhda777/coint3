import pandas as pd  # type: ignore
from pathlib import Path
from typing import List


class DataHandler:
    """Utility class for loading local parquet price files."""

    def __init__(self, data_dir: Path, timeframe: str, fill_limit_pct: float) -> None:
        self.data_dir = Path(data_dir)
        # timeframe kept for metadata compatibility but no longer used in path
        self.timeframe = timeframe
        self.fill_limit_pct = fill_limit_pct
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._all_data_cache: pd.DataFrame | None = None

    def get_all_symbols(self) -> List[str]:
        """Return list of symbols based on partition directory names."""
        if not self.data_dir.exists():
            return []

        symbols = []
        for path in self.data_dir.iterdir():
            if path.is_dir() and path.name.startswith("symbol="):
                symbols.append(path.name.replace("symbol=", ""))
        return sorted(symbols)

    def _load_full_dataset(self) -> pd.DataFrame:
        """Load the entire partitioned dataset and cache the result."""
        if self._all_data_cache is not None:
            return self._all_data_cache

        if not self.data_dir.exists():
            self._all_data_cache = pd.DataFrame()
            return self._all_data_cache

        full_df = pd.read_parquet(self.data_dir, engine="pyarrow")

        if "timestamp" in full_df.columns:
            full_df = full_df.set_index("timestamp")

        full_df.index = pd.to_datetime(full_df.index)
        full_df = full_df.sort_index()

        self._all_data_cache = full_df
        return full_df

    def load_all_data_for_period(self, lookback_days: int) -> pd.DataFrame:
        """Load close prices for all symbols for the given lookback period."""
        full_df = self._load_full_dataset()
        if full_df.empty:
            return pd.DataFrame()

        end_date = full_df.index.max()
        start_date = end_date - pd.Timedelta(days=lookback_days)

        filtered_df = full_df[full_df.index >= start_date]
        wide = filtered_df.pivot_table(index=filtered_df.index, columns="symbol", values="close")
        return wide

    def load_pair_data(self, symbol1: str, symbol2: str) -> pd.DataFrame:
        """Load and align data for two symbols."""
        full_df = self._load_full_dataset()
        if full_df.empty:
            return pd.DataFrame()

        pair_df = full_df[full_df["symbol"].isin([symbol1, symbol2])]
        wide_df = pair_df.pivot_table(index=pair_df.index, columns="symbol", values="close")

        if wide_df.empty:
            return pd.DataFrame()

        freq = pd.infer_freq(wide_df.index) or "D"
        wide_df = wide_df.asfreq(freq)
        limit = int(len(wide_df) * self.fill_limit_pct)
        wide_df = wide_df.ffill(limit=limit).bfill(limit=limit)

        return wide_df[[symbol1, symbol2]].dropna()
