import logging
import threading
import time
from pathlib import Path

import dask.dataframe as dd
import pandas as pd  # type: ignore
import pyarrow as pa
import pyarrow.dataset as ds

from coint2.utils import empty_ddf, ensure_datetime_index
from coint2.utils.config import AppConfig

# Настройка логгера
logger = logging.getLogger(__name__)


def _scan_parquet_files(path: Path | str, glob: str = "*.parquet", max_shards: int | None = None) -> ds.Dataset:
    """Build pyarrow dataset from parquet files under ``path``."""
    base = Path(path)
    files = sorted(base.rglob(glob))
    if max_shards is not None:
        files = files[:max_shards]
    if not files:
        return ds.dataset(pa.table({}))
    return ds.dataset([str(f) for f in files], format="parquet", partitioning="hive")


def _dir_mtime_hash(path: Path) -> float:
    """Return hash value based on modification times of ``.parquet`` files."""
    mtimes = [f.stat().st_mtime for f in path.rglob("*.parquet")]
    return max(mtimes) if mtimes else 0.0


class DataHandler:
    """Utility class for loading local parquet price files."""

    def __init__(self, cfg: AppConfig, autorefresh: bool = True) -> None:
        self.data_dir = Path(cfg.data_dir)
        self.fill_limit_pct = cfg.backtest.fill_limit_pct
        self.max_shards = cfg.max_shards
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.autorefresh = autorefresh
        self._all_data_cache: dict[str, tuple[dd.DataFrame, float]] = {}
        self._freq: str | None = None
        self._lock = threading.Lock()
        self.lookback_days: int = cfg.pair_selection.lookback_days

    @property
    def freq(self) -> str | None:
        """Return detected time step of the loaded data."""
        with self._lock:
            return self._freq

    def clear_cache(self) -> None:
        """Clears the in-memory Dask DataFrame cache."""
        with self._lock:
            self._all_data_cache.clear()
            self._freq = None

    def get_all_symbols(self) -> list[str]:
        """Return list of symbols based on partition directory names."""
        if not self.data_dir.exists():
            return []

        symbols = []
        for path in self.data_dir.iterdir():
            if path.is_dir() and path.name.startswith("symbol="):
                symbols.append(path.name.replace("symbol=", ""))
        return sorted(symbols)

    def _load_full_dataset(self) -> dd.DataFrame:
        """Load the full dataset using Dask.
        
        Returns:
        -------
        dask.dataframe.DataFrame
            Lazy Dask DataFrame with the data.
        """
        current_hash = _dir_mtime_hash(self.data_dir) if self.autorefresh else 0.0

        with self._lock:
            cached_entry = self._all_data_cache.get("all")

        if cached_entry is not None:
            cached_ddf, cached_hash = cached_entry
            if not self.autorefresh or cached_hash == current_hash:
                return cached_ddf

        if not self.data_dir.exists():
            empty = empty_ddf()
            with self._lock:
                self._all_data_cache["all"] = (empty, current_hash)
            return empty

        try:
            # Использование dd.read_parquet с правильными настройками
            ddf = dd.read_parquet(
                self.data_dir,
                engine="pyarrow",
                columns=None,  # partition columns like 'symbol' included automatically
                ignore_metadata_file=True,  # Игнорируем _metadata файл
                calculate_divisions=False,   # Избегаем авто-расчёта разделений
                # В партициях используется ведущий ноль для месяцев (month=01, month=02 и т.д.)
                filters=None,  # Не применяем фильтры на уровне партиций
                validate_schema=False,  # Отключаем строгую валидацию схемы
            )
            with self._lock:
                self._all_data_cache["all"] = (ddf, current_hash)
            return ddf
        except Exception as e:
            logger.debug(f"Ошибка загрузки данных через Dask в _load_full_dataset: {str(e)}")

            try:
                logger.debug("Fallback to pyarrow dataset scanning")
                dataset = _scan_parquet_files(self.data_dir, max_shards=self.max_shards)
                table = dataset.to_table(columns=["timestamp", "close", "symbol"])
                pdf = table.to_pandas()
                if pdf.empty:
                    empty = empty_ddf()
                    with self._lock:
                        self._all_data_cache["all"] = (empty, current_hash)
                    return empty

                npartitions = max(1, min(32, len(pdf) // 100000 + 1))
                ddf = dd.from_pandas(pdf, npartitions=npartitions)
                with self._lock:
                    self._all_data_cache["all"] = (ddf, current_hash)
                return ddf
            except Exception as e2:
                logger.debug(f"Ошибка при использовании запасного варианта: {str(e2)}")
                empty = empty_ddf()
                with self._lock:
                    self._all_data_cache["all"] = (empty, current_hash)
                return empty

    def load_all_data_for_period(self, lookback_days: int | None = None) -> pd.DataFrame:
        """Load close prices for all symbols for the specified or configured lookback period.
        
        Parameters
        ----------
        lookback_days : int | None, optional
            Number of days to look back. If None, uses self.lookback_days.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with close prices for all symbols.
        """
        try:
            ddf = self._load_full_dataset()

            # Проверка на пустой DataFrame
            if not ddf.columns.tolist():
                return pd.DataFrame()

            # Конвертируем timestamp в datetime
            ddf["timestamp"] = dd.to_datetime(ddf["timestamp"])
            
            # Находим максимальную дату
            end_date = ddf["timestamp"].max().compute()
            if pd.isna(end_date):
                return pd.DataFrame()

            # Используем переданный параметр или значение по умолчанию
            days_to_lookback = lookback_days if lookback_days is not None else self.lookback_days
            
            # Вычисляем начальную дату для фильтрации
            start_date = end_date - pd.Timedelta(days=days_to_lookback)
            
            # Фильтруем по дате
            filtered_ddf = ddf[ddf["timestamp"] >= start_date]

            # Вычисляем отфильтрованные данные и выполняем pivot уже в pandas
            filtered_df = filtered_ddf.compute()
            if filtered_df.empty:
                return pd.DataFrame()

            if filtered_df.duplicated(subset=["timestamp", "symbol"]).any():
                wide_pdf = filtered_df.pivot_table(
                    index="timestamp",
                    columns="symbol",
                    values="close",
                    aggfunc="last",
                )
            else:
                wide_pdf = filtered_df.pivot(
                    index="timestamp",
                    columns="symbol",
                    values="close",
                )
            if wide_pdf.empty:
                return pd.DataFrame()

            wide_pdf = ensure_datetime_index(wide_pdf)

            freq_val = pd.infer_freq(wide_pdf.index)
            with self._lock:
                self._freq = freq_val
            if freq_val:
                wide_pdf = wide_pdf.asfreq(freq_val)

            return wide_pdf
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных за период: {str(e)}")
            return pd.DataFrame()

    def load_pair_data(
        self,
        symbol1: str,
        symbol2: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Load and align data for two symbols within the given date range."""
        # Обеспечиваем, что даты в наивном формате (без timezone)
        if start_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из start_date: {start_date}")
            start_date = start_date.tz_localize(None)
        if end_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из end_date: {end_date}")
            end_date = end_date.tz_localize(None)
            
        logger.debug(
            f"Загрузка данных для пары {symbol1}-{symbol2} ({start_date} - {end_date})"
        )

        if not self.data_dir.exists():
            logger.warning(f"Директория данных {self.data_dir} не существует")
            return pd.DataFrame()

        filters = [
            ("symbol", "in", [symbol1, symbol2]),
            ("timestamp", ">=", start_date),
            ("timestamp", "<=", end_date),
        ]

        try:
            ddf = dd.read_parquet(
                self.data_dir,
                engine="pyarrow",
                columns=["timestamp", "close", "symbol"],
                ignore_metadata_file=True,
                calculate_divisions=False,
                filters=filters,
                validate_schema=False,
            )
            pair_pdf = ddf.compute()
        except Exception as e:  # pragma: no cover - fallback rarely used
            logger.debug(f"Error loading pair data via Dask: {str(e)}")
            try:
                logger.debug("Fallback to pyarrow dataset scanning in load_pair_data")
                dataset = _scan_parquet_files(self.data_dir, max_shards=self.max_shards)
                arrow_filter = (
                    ds.field("symbol").isin([symbol1, symbol2])
                    & (ds.field("timestamp") >= pa.scalar(start_date))
                    & (ds.field("timestamp") <= pa.scalar(end_date))
                )
                table = dataset.to_table(
                    columns=["timestamp", "close", "symbol"],
                    filter=arrow_filter,
                )
                pair_pdf = table.to_pandas()
            except Exception as e2:  # pragma: no cover - fallback rarely used
                logger.debug(f"Error loading pair data manually: {str(e2)}")
                return pd.DataFrame()

        if pair_pdf.empty:
            logger.debug(f"Нет данных для пары {symbol1}-{symbol2}")
            return pd.DataFrame()
        pair_pdf["timestamp"] = pd.to_datetime(pair_pdf["timestamp"]).dt.tz_localize(None)

        # Проверка на наличие дубликатов timestamp
        if pair_pdf.duplicated(subset=["timestamp", "symbol"]).any():
            logger.debug(f"Обнаружены дубликаты timestamp для пары {symbol1}-{symbol2}. Удаляем дубликаты.")
            pair_pdf = pair_pdf.drop_duplicates(subset=["timestamp", "symbol"])

        # Преобразуем в широкий формат (timestamp x symbols)
        wide_df = pair_pdf.pivot_table(index="timestamp", columns="symbol", values="close")

        # Проверка на наличие нужных столбцов
        if wide_df.empty or len(wide_df.columns) < 2:
            return pd.DataFrame()

        # Обработка пропущенных значений
        freq_val = pd.infer_freq(wide_df.index)
        with self._lock:
            self._freq = freq_val
        if freq_val:
            wide_df = wide_df.asfreq(freq_val)

        # рассчитываем максимальную длину подряд идущих NA по проценту
        limit = int(len(wide_df) * self.fill_limit_pct)

        wide_df = wide_df.ffill(limit=limit).bfill(limit=limit)


        # Возвращаем только нужные символы и удаляем строки с NA
        if symbol1 in wide_df.columns and symbol2 in wide_df.columns:
            return wide_df[[symbol1, symbol2]].dropna()
        else:
            return pd.DataFrame()

    def load_and_normalize_data(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Load and normalize data for all symbols within the given date range.
        
        Parameters
        ----------
        start_date : pd.Timestamp
            Начальная дата для загрузки данных
        end_date : pd.Timestamp
            Конечная дата для загрузки данных
            
        Returns
        -------
        pd.DataFrame
            Нормализованный DataFrame с ценами (начало = 100)
        """
        # Обеспечиваем, что даты в наивном формате (без timezone)
        if start_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из start_date: {start_date}")
            start_date = start_date.tz_localize(None)
        if end_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из end_date: {end_date}")
            end_date = end_date.tz_localize(None)
            
        logger.debug(f"Загрузка и нормализация данных за период {start_date} - {end_date}")
        
        start_time = time.time()
        data_df = self.preload_all_data(start_date, end_date)
        elapsed_time = time.time() - start_time
        logger.info(f"Данные загружены за {elapsed_time:.2f} секунд. Размер: {data_df.shape}")

        # Нормализация цен
        if not data_df.empty:
            logger.debug(
                f"Количество символов до нормализации: {len(data_df.columns)}"
            )

            numeric_cols = [
                c
                for c in data_df.columns
                if pd.api.types.is_numeric_dtype(data_df[c]) and c != "timestamp"
            ]
            if numeric_cols:
                first_values = data_df[numeric_cols].apply(
                    lambda s: s.loc[s.first_valid_index()] if s.first_valid_index() is not None else pd.NA
                )
                valid_cols = first_values[(first_values != 0) & first_values.notna()].index
                if len(valid_cols) > 0:
                    data_df[valid_cols] = (
                        100 * data_df[valid_cols].div(first_values.loc[valid_cols])
                    )

            logger.debug(
                f"Количество символов после нормализации: {len(data_df.columns)}"
            )
            
            # Проверяем наличие константных серий и серий с большим количеством пропусков
            valid_columns = []
            for column in data_df.columns:
                if pd.api.types.is_numeric_dtype(data_df[column]):
                    # Проверка на константность серии
                    if data_df[column].nunique() > 1:
                        # Проверка на слишком много пропусков (более 50%)
                        na_pct = data_df[column].isna().mean()
                        if na_pct < 0.5:
                            valid_columns.append(column)
                            
            # Оставляем только валидные столбцы
            if valid_columns:
                logger.debug(
                    f"Отфильтровано {len(data_df.columns) - len(valid_columns)} "
                    "константных или разреженных серий"
                )
                data_df = data_df[valid_columns]

        return data_df

    def preload_all_data(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Loads and pivots all data for a given wide date range."""
        # Обеспечиваем, что даты в наивном формате (без timezone)
        if start_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из start_date в preload_all_data: {start_date}")
            start_date = start_date.tz_localize(None)
        if end_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из end_date в preload_all_data: {end_date}")
            end_date = end_date.tz_localize(None)
            
        logger.debug(f"Предзагрузка всех данных за период {start_date} - {end_date}")
            
        if not self.data_dir.exists():
            logger.warning(f"Директория данных {self.data_dir} не существует")
            return pd.DataFrame()

        # Преобразуем даты в timestamp формат для фильтрации
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        logger.debug(f"Фильтрация по timestamp: {start_ts} - {end_ts}")
        
        filters = [
            ("timestamp", ">=", start_ts),
            ("timestamp", "<=", end_ts),
        ]

        try:
            # Для партиционированных данных лучше загружать без фильтров и фильтровать после
            logger.debug("Загрузка всех данных без фильтров...")
            ddf = dd.read_parquet(
                self.data_dir,
                engine="pyarrow",
                columns=["timestamp", "close", "symbol"],
                ignore_metadata_file=True,
                calculate_divisions=False,
                validate_schema=False,
            )

            # Преобразуем в pandas для фильтрации
            all_data = ddf.compute()
            logger.debug(f"Загружено записей всего: {len(all_data)}")
            
            if all_data.empty:
                logger.warning(f"Нет данных в директории {self.data_dir}")
                return pd.DataFrame()
            
            # Фильтруем по времени в pandas
            mask = (all_data["timestamp"] >= start_ts) & (all_data["timestamp"] <= end_ts)
            filtered_df = all_data[mask]
            logger.debug(f"Записей после фильтрации по времени: {len(filtered_df)}")
            
            if filtered_df.empty:
                logger.warning(f"No data found between {start_date} and {end_date}")
                logger.debug(f"Доступный диапазон времени: {pd.to_datetime(all_data['timestamp'].min(), unit='ms')} - {pd.to_datetime(all_data['timestamp'].max(), unit='ms')}")
                return pd.DataFrame()

            # Преобразуем timestamp в datetime для индекса
            filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"], unit="ms")
            
            if filtered_df.duplicated(subset=["timestamp", "symbol"]).any():
                wide_pdf = filtered_df.pivot_table(
                    index="timestamp",
                    columns="symbol",
                    values="close",
                    aggfunc="last",
                )
            else:
                wide_pdf = filtered_df.pivot(
                    index="timestamp",
                    columns="symbol",
                    values="close",
                )

            if wide_pdf.empty:
                logger.debug(f"Пустой pivot после преобразования")
                return pd.DataFrame()

            wide_pdf = ensure_datetime_index(wide_pdf)
            logger.debug(f"Итоговые данны: {wide_pdf.shape}, период: {wide_pdf.index.min()} - {wide_pdf.index.max()}")

            freq_val = pd.infer_freq(wide_pdf.index)
            with self._lock:
                self._freq = freq_val
            if freq_val:
                wide_pdf = wide_pdf.asfreq(freq_val)

            return wide_pdf
        except Exception as e:
            logger.error(f"Error loading data via Dask: {str(e)}")

            try:
                logger.debug("Fallback to pyarrow dataset scanning in preload_all_data")
                dataset = _scan_parquet_files(self.data_dir, max_shards=self.max_shards)
                table = dataset.to_table(columns=["timestamp", "close", "symbol"])
                pdf = table.to_pandas()
                if pdf.empty:
                    return pd.DataFrame()

                # Фильтруем по timestamp (в миллисекундах)
                mask = (pdf["timestamp"] >= start_ts) & (pdf["timestamp"] <= end_ts)
                pdf = pdf.loc[mask]
                if pdf.empty:
                    logger.debug(f"Нет данных в диапазоне {start_ts} - {end_ts}")
                    return pd.DataFrame()

                # Преобразуем timestamp в datetime
                pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], unit="ms")
                
                wide_df = pdf.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
                wide_df = ensure_datetime_index(wide_df)
                freq_val = pd.infer_freq(wide_df.index)
                with self._lock:
                    self._freq = freq_val
                if freq_val:
                    wide_df = wide_df.asfreq(freq_val)
                logger.debug(f"Successfully loaded data via fallback. Shape: {wide_df.shape}")
                return wide_df
            except Exception as e2:
                logger.error(f"Error loading data manually: {str(e2)}")
                return pd.DataFrame()
