import pandas as pd  # type: ignore
from pathlib import Path
from typing import List


class DataHandler:
    """Utility class for loading local parquet price files."""

    def __init__(self, data_dir: Path, timeframe: str, fill_limit_pct: float) -> None:
        self.data_dir = Path(data_dir) / timeframe
        self.fill_limit_pct = fill_limit_pct
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_all_symbols(self) -> List[str]:
        """Return list of symbols based on parquet file names."""
        return [p.stem for p in sorted(self.data_dir.glob("*.parquet"))]

    def _load_symbol_data(self, symbol: str) -> pd.DataFrame:
        path = self.data_dir / f"{symbol}.parquet"
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df[["close"]].rename(columns={"close": symbol})

    def load_all_data_for_period(self, lookback_days: int) -> pd.DataFrame:
        """Load close prices for all symbols for the given lookback period."""
        all_symbols = self.get_all_symbols()
        dfs = []
        for sym in all_symbols:
            df = self._load_symbol_data(sym)
            if lookback_days:
                end = df.index.max()
                start = end - pd.Timedelta(days=lookback_days)
                df = df.loc[df.index >= start]
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, axis=1).sort_index()

    def load_pair_data(self, symbol1: str, symbol2: str) -> pd.DataFrame:
        """Load and align data for two symbols."""
        df1 = self._load_symbol_data(symbol1)
        df2 = self._load_symbol_data(symbol2)
        df = pd.concat([df1, df2], axis=1)
        limit = int(len(df) * self.fill_limit_pct)
        df = df.sort_index()
        df = df.ffill(limit=limit).bfill(limit=limit)
        return df.dropna()