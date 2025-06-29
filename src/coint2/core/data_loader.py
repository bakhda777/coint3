import dask.dataframe as dd
import pandas as pd  # type: ignore
from pathlib import Path
from typing import List
import threading
import time
import numpy as np
import logging
import pyarrow.dataset as ds
import pyarrow as pa

from coint2.utils.config import AppConfig
from coint2.utils import empty_ddf, ensure_datetime_index, infer_frequency

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


class DataHandler:
    """Utility class for loading local parquet price files."""

    def __init__(self, cfg: AppConfig) -> None:
        self.data_dir = Path(cfg.data_dir)
        self.fill_limit_pct = cfg.backtest.fill_limit_pct
        self.max_shards = cfg.max_shards
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._all_data_cache: dd.DataFrame | None = None
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
            self._all_data_cache = None
            self._freq = None

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
        """Load the full dataset using Dask.
        
        Returns:
        -------
        dask.dataframe.DataFrame
            Lazy Dask DataFrame with the data.
        """
        with self._lock:
            cached = self._all_data_cache
        if cached is not None:
            return cached

        if not self.data_dir.exists():
            with self._lock:
                self._all_data_cache = empty_ddf()
                return self._all_data_cache

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
                self._all_data_cache = ddf
            return ddf
        except Exception as e:
            logger.debug(f"Ошибка загрузки данных через Dask в _load_full_dataset: {str(e)}")

            try:
                logger.debug("Fallback to pyarrow dataset scanning")
                dataset = _scan_parquet_files(self.data_dir, max_shards=self.max_shards)
                table = dataset.to_table(columns=["timestamp", "close", "symbol"])
                pdf = table.to_pandas()
                if pdf.empty:
                    with self._lock:
                        self._all_data_cache = empty_ddf()
                    return self._all_data_cache

                npartitions = max(1, min(32, len(pdf) // 100000 + 1))
                ddf = dd.from_pandas(pdf, npartitions=npartitions)
                with self._lock:
                    self._all_data_cache = ddf
                return ddf
            except Exception as e2:
                logger.debug(f"Ошибка при использовании запасного варианта: {str(e2)}")
                with self._lock:
                    self._all_data_cache = empty_ddf()
                return self._all_data_cache

    def load_all_data_for_period(self) -> pd.DataFrame:
        """Load close prices for all symbols for the configured lookback period."""

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

        # Вычисляем начальную дату для фильтрации
        start_date = end_date - pd.Timedelta(days=self.lookback_days)
        
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
            
        logger.debug(f"Загрузка данных для пары {symbol1}-{symbol2} ({start_date} - {end_date})")
        
        ddf = self._load_full_dataset()

        # Проверка на пустой DataFrame
        if not ddf.columns.tolist():
            logger.debug(f"Пустой DataFrame для пары {symbol1}-{symbol2}")
            return pd.DataFrame()
            
        # Фильтруем только интересующие нас символы
        pair_ddf = ddf[ddf["symbol"].isin([symbol1, symbol2])]

        # Конвертируем в pandas для дальнейшей обработки
        pair_pdf = pair_ddf.compute()

        # Проверка на пустой результат
        if pair_pdf.empty:
            logger.debug(f"Нет данных для пары {symbol1}-{symbol2}")
            return pd.DataFrame()

        # Убедимся, что timestamp в формате datetime и без timezone
        pair_pdf["timestamp"] = pd.to_datetime(pair_pdf["timestamp"])
        
        # Удаляем timezone, если она есть
        if hasattr(pair_pdf["timestamp"].dt, 'tz') and pair_pdf["timestamp"].dt.tz is not None:
            logger.debug(f"Удаляем timezone из данных (была {pair_pdf['timestamp'].dt.tz})")
            pair_pdf["timestamp"] = pair_pdf["timestamp"].dt.tz_localize(None)
        
        # Фильтруем по диапазону дат (теперь все наивные даты)
        mask = (pair_pdf["timestamp"] >= start_date) & (pair_pdf["timestamp"] <= end_date)
        pair_pdf = pair_pdf.loc[mask]

        # Проверка на пустой результат после фильтрации
        if pair_pdf.empty:
            logger.debug(f"Нет данных для пары {symbol1}-{symbol2} после фильтрации по дате")
            return pd.DataFrame()

        # Проверка на наличие дубликатов timestamp
        if pair_pdf.duplicated(subset=["timestamp", "symbol"]).any():
            logger.debug(f"Обнаружены дубликаты timestamp для пары {symbol1}-{symbol2}. Удаляем дубликаты.")
            pair_pdf = pair_pdf.drop_duplicates(subset=["timestamp", "symbol"])

        # Рассчитываем допустимое количество последовательных пропусков до группировки
        total_rows = pair_pdf["timestamp"].nunique()
        limit = int(total_rows * self.fill_limit_pct)

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
            logger.debug(f"Количество символов до нормализации: {len(data_df.columns)}")
            for column in data_df.columns:
                if pd.api.types.is_numeric_dtype(data_df[column]) and column != "timestamp":
                    # Нормализуем к начальному значению 100
                    first_valid_idx = data_df[column].first_valid_index()
                    if first_valid_idx is not None and not pd.isna(data_df.loc[first_valid_idx, column]):
                        first_value = data_df.loc[first_valid_idx, column]
                        if first_value != 0:
                            data_df[column] = 100 * data_df[column] / first_value
            logger.debug(f"Количество символов после нормализации: {len(data_df.columns)}")
            
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
                logger.debug(f"Отфильтровано {len(data_df.columns) - len(valid_columns)} константных или разреженных серий")
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

        try:
            # Загружаем данные через Dask с оптимизированными параметрами
            ddf = self._load_full_dataset()

            # Проверка на пустой DataFrame
            if not ddf.columns.tolist():
                logger.warning("No columns found in dataset")
                return pd.DataFrame()

            # Конвертируем timestamp в datetime БЕЗ timezone
            ddf["timestamp"] = dd.to_datetime(ddf["timestamp"])
            
            # Удаляем timezone из timestamp (если есть)
            try:
                if hasattr(ddf["timestamp"], 'dt') and ddf["timestamp"].dt.tz is not None:
                    logger.debug("Удаляем timezone из данных в Dask DF")
                    ddf["timestamp"] = ddf["timestamp"].dt.tz_localize(None)
            except (AttributeError, TypeError) as e:
                logger.debug(f"Не удалось проверить timezone в Dask DataFrame: {str(e)}")
            
            # Фильтруем данные по заданному диапазону дат (теперь все наивные даты)
            filtered_ddf = ddf[
                (ddf["timestamp"] >= start_date) & 
                (ddf["timestamp"] <= end_date)
            ]

            # Вычисляем отфильтрованные данные и выполняем pivot уже в pandas
            filtered_df = filtered_ddf.compute()
            if filtered_df.empty:
                logger.debug(f"No data found between {start_date} and {end_date}")
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
                logger.debug(f"No data found between {start_date} and {end_date}")
                return pd.DataFrame()

            wide_pdf = ensure_datetime_index(wide_pdf)

            freq_val = pd.infer_freq(wide_pdf.index)
            with self._lock:
                self._freq = freq_val
            if freq_val:
                wide_pdf = wide_pdf.asfreq(freq_val)

            return wide_pdf
        except Exception as e:
            logger.debug(f"Error loading data via Dask: {str(e)}")

            try:
                logger.debug("Fallback to pyarrow dataset scanning in preload_all_data")
                dataset = _scan_parquet_files(self.data_dir, max_shards=self.max_shards)
                table = dataset.to_table(columns=["timestamp", "close", "symbol"])
                pdf = table.to_pandas()
                if pdf.empty:
                    return pd.DataFrame()

                start_date_naive = start_date.tz_localize(None) if start_date.tzinfo is not None else start_date
                end_date_naive = end_date.tz_localize(None) if end_date.tzinfo is not None else end_date
                mask = (pdf["timestamp"] >= start_date_naive) & (pdf["timestamp"] <= end_date_naive)
                pdf = pdf.loc[mask]
                if pdf.empty:
                    return pd.DataFrame()

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
                logger.debug(f"Error loading data manually: {str(e2)}")
                return pd.DataFrame()
