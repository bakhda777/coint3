import dask.dataframe as dd
import pandas as pd  # type: ignore
from pathlib import Path
from typing import List
import time
import numpy as np
import logging

from coint2.utils.config import AppConfig
from coint2.utils import empty_ddf

# Настройка логгера
logger = logging.getLogger(__name__)


class DataHandler:
    """Utility class for loading local parquet price files."""

    def __init__(self, cfg: AppConfig) -> None:
        self.data_dir = Path(cfg.data_dir)
        self.fill_limit_pct = cfg.backtest.fill_limit_pct
        self.max_shards = cfg.max_shards
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._all_data_cache: dd.DataFrame | None = None
        self._freq: str | None = None

    @property
    def freq(self) -> str | None:
        """Return detected time step of the loaded data."""
        return self._freq

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
        """Load the full dataset using Dask.
        
        Returns:
        -------
        dask.dataframe.DataFrame
            Lazy Dask DataFrame with the data.
        """
        if self._all_data_cache is not None:
            return self._all_data_cache

        if not self.data_dir.exists():
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
            self._all_data_cache = ddf
            return ddf
        except Exception as e:
            print(f"Ошибка загрузки данных через Dask в _load_full_dataset: {str(e)}")
            
            try:
                # Запасной вариант: ручной обход всех parquet-файлов
                print("Попытка загрузки через glob-обход файлов...")

                # Собираем все файлы .parquet рекурсивно
                parquet_files = [str(p) for p in Path(self.data_dir).rglob("*.parquet")]
                if not parquet_files:
                    print(f"Не найдено ни одного parquet файла в {self.data_dir}")

                    return empty_ddf()
                
                print(f"Найдено {len(parquet_files)} файлов parquet")

                if self.max_shards is not None:
                    parquet_files = parquet_files[: self.max_shards]
                
                # Словарь для отслеживания проанализированных символов для диагностики
                analyzed_symbols = set()
                
                # Создаем список Dask DataFrame из каждого файла
                dfs = []
                for file_path in parquet_files:
                    path = Path(file_path)
                    
                    # Определяем symbol из пути (3 уровня вверх от файла: symbol=XXX/year=YYYY/month=MM/part.parquet)
                    symbol_dir = path.parent.parent.parent.name
                    symbol = symbol_dir.replace("symbol=", "")
                    
                    try:
                        # Читаем файл напрямую через pandas с отключением проверки схемы
                        # и использованием только pyarrow без dask
                        file_df = pd.read_parquet(
                            file_path, 
                            engine="pyarrow",
                            columns=["timestamp", "close"]
                        )
                        
                        # Диагностическая информация о формате timestamp для первого файла каждого символа
                        # Извлекаем имя символа из пути
                        symbol_dir = path.parent.parent.parent.name
                        current_symbol = symbol_dir.replace("symbol=", "")
                        
                        # Проверяем, не анализировали ли мы уже этот символ и является ли дата в нужном диапазоне
                        if not file_df.empty and current_symbol not in analyzed_symbols and pd.Timestamp('2022-01-01') <= file_df['timestamp'].min() <= pd.Timestamp('2022-12-31'):
                            analyzed_symbols.add(current_symbol)
                            print(f"\nDEBUG: Анализ формата данных для символа {current_symbol}")
                            print(f"DEBUG: Файл {file_path}")
                            print(f"DEBUG: Тип timestamp: {type(file_df['timestamp'].iloc[0])}")
                            print(f"DEBUG: Первые timestamp: {file_df['timestamp'].head(3).tolist()}")
                            
                            # Проверка 15-минутных интервалов
                            if len(file_df) >= 10:
                                sorted_ts = file_df['timestamp'].sort_values().reset_index(drop=True)
                                print(f"DEBUG: Детальные timestamp (первые 10): {sorted_ts[:10].tolist()}")
                                
                                # Проверяем разницу между соседними метками времени
                                diffs = []
                                for i in range(1, min(10, len(sorted_ts))):
                                    diff = (sorted_ts.iloc[i] - sorted_ts.iloc[i-1]).total_seconds() / 60
                                    diffs.append(diff)
                                print(f"DEBUG: Интервалы между записями (в минутах): {diffs}")
                                print(f"DEBUG: Часы для первых 10 записей: {[ts.hour for ts in sorted_ts[:min(10, len(sorted_ts))]]}")
                                print(f"DEBUG: Минуты для первых 10 записей: {[ts.minute for ts in sorted_ts[:min(10, len(sorted_ts))]]}")
                                
                                # Хистограмма часов и минут
                                hours_count = file_df['timestamp'].dt.hour.value_counts().sort_index()
                                minutes_count = file_df['timestamp'].dt.minute.value_counts().sort_index()
                                print(f"DEBUG: Уникальные значения часов: {list(hours_count.index)}")
                                print(f"DEBUG: Уникальные значения минут: {list(minutes_count.index)}")
                                
                                # Проверка на 15-минутные интервалы
                                expected_minutes = [0, 15, 30, 45]
                                is_15min = all(minute in expected_minutes for minute in minutes_count.index)
                                print(f"DEBUG: Данные соответствуют 15-минутным интервалам: {is_15min}")
                            else:
                                print("DEBUG: Файл содержит слишком мало записей для анализа интервалов")
                        
                        # Преобразуем timestamp в datetime для всех файлов, если это число
                        if not file_df.empty and isinstance(file_df['timestamp'].iloc[0], (int, float, np.integer, np.floating)):
                            # Если это unix timestamp в миллисекундах
                            if file_df['timestamp'].iloc[0] > 1e12:  # Больше 2001 года в миллисекундах
                                file_df['timestamp'] = pd.to_datetime(file_df['timestamp'], unit='ms')
                                if path.parent.parent.parent.name == "symbol=BTCUSDT" and path.parent.parent.name == "year=2022" and path.parent.name == "month=01":
                                    print(f"DEBUG: Преобразован timestamp из миллисекунд: {file_df['timestamp'].head(3)}")
                            else:
                                file_df['timestamp'] = pd.to_datetime(file_df['timestamp'], unit='s')
                                if path.parent.parent.parent.name == "symbol=BTCUSDT" and path.parent.parent.name == "year=2022" and path.parent.name == "month=01":
                                    print(f"DEBUG: Преобразован timestamp из секунд: {file_df['timestamp'].head(3)}")
                        
                        # Убедимся, что все timestamp в наивном формате (без timezone)
                        if not file_df.empty and isinstance(file_df['timestamp'].iloc[0], pd.Timestamp):
                            # Если есть timezone, убираем его
                            if file_df['timestamp'].dt.tz is not None:
                                file_df['timestamp'] = file_df['timestamp'].dt.tz_localize(None)
                        
                        # Добавляем информацию о символе
                        file_df["symbol"] = symbol
                        
                        # Создаем маленький Dask DataFrame из pandas DataFrame
                        file_ddf = dd.from_pandas(file_df, npartitions=1)
                        dfs.append(file_ddf)
                    except Exception as file_error:
                        print(f"Ошибка при чтении файла {file_path}: {str(file_error)}")
                        continue
                    
                if not dfs:
                    print("Не удалось загрузить ни один файл")
                    self._all_data_cache = dd.from_pandas(pd.DataFrame(), npartitions=1)
                    return self._all_data_cache
                    
                try:
                    # Объединяем все Dask DataFrame в один
                    combined_ddf = dd.concat(dfs)
                    print("Успешно создан объединенный Dask DataFrame")
                    self._all_data_cache = combined_ddf
                    return combined_ddf
                except Exception as concat_error:
                    print(f"Ошибка при объединении DataFrame: {str(concat_error)}")
                    self._all_data_cache = dd.from_pandas(pd.DataFrame(), npartitions=1)
                    return self._all_data_cache
                    
            except Exception as e2:
                print(f"Ошибка при использовании запасного варианта: {str(e2)}")
                # Если и запасной вариант не сработал, возвращаем пустой фрейм
                self._all_data_cache = dd.from_pandas(pd.DataFrame(), npartitions=1)
                return self._all_data_cache

    def load_all_data_for_period(self, lookback_days: int) -> pd.DataFrame:
        """Load close prices for all symbols for the given lookback period."""
        ddf = self._load_full_dataset()

        # Проверка на пустой DataFrame
        if not ddf.columns.compute().tolist():
            return pd.DataFrame()

        # Конвертируем timestamp в datetime
        ddf["timestamp"] = dd.to_datetime(ddf["timestamp"])
        
        # Находим максимальную дату
        end_date = ddf["timestamp"].max().compute()
        if pd.isna(end_date):
            return pd.DataFrame()

        # Вычисляем начальную дату для фильтрации
        start_date = end_date - pd.Timedelta(days=lookback_days)
        
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

        # Сортируем по индексу (дате)
        wide_pdf = wide_pdf.sort_index()

        self._freq = pd.infer_freq(wide_pdf.index)
        if self._freq:
            wide_pdf = wide_pdf.asfreq(self._freq)

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
        if not ddf.columns.compute().tolist():
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
            print(f"Обнаружены дубликаты timestamp для пары {symbol1}-{symbol2}. Удаляем дубликаты.")
            pair_pdf = pair_pdf.drop_duplicates(subset=["timestamp", "symbol"])

        # Преобразуем в широкий формат (timestamp x symbols)
        wide_df = pair_pdf.pivot_table(index="timestamp", columns="symbol", values="close")

        # Проверка на наличие нужных столбцов
        if wide_df.empty or len(wide_df.columns) < 2:
            return pd.DataFrame()

        # Обработка пропущенных значений
        freq = pd.infer_freq(wide_df.index)
        self._freq = freq
        if freq:
            wide_df = wide_df.asfreq(freq)
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
            print(f"Количество символов до нормализации: {len(data_df.columns)}")
            for column in data_df.columns:
                if pd.api.types.is_numeric_dtype(data_df[column]) and column != "timestamp":
                    # Нормализуем к начальному значению 100
                    first_valid_idx = data_df[column].first_valid_index()
                    if first_valid_idx is not None and not pd.isna(data_df.loc[first_valid_idx, column]):
                        first_value = data_df.loc[first_valid_idx, column]
                        if first_value != 0:
                            data_df[column] = 100 * data_df[column] / first_value
            print(f"Количество символов после нормализации: {len(data_df.columns)}")
            
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
                print(f"Отфильтровано {len(data_df.columns) - len(valid_columns)} константных или разреженных серий")
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
            if not ddf.columns.compute().tolist():
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
                print(f"No data found between {start_date} and {end_date}")
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
                print(f"No data found between {start_date} and {end_date}")
                return pd.DataFrame()

            # Сортируем по индексу (датам)
            wide_pdf = wide_pdf.sort_index()

            self._freq = pd.infer_freq(wide_pdf.index)
            if self._freq:
                wide_pdf = wide_pdf.asfreq(self._freq)

            return wide_pdf
            
        except Exception as e:
            # Если Dask-подход не сработал, переходим на ручной обход через pandas
            print(f"Error loading data via Dask: {str(e)}")
            
            try:
                # Запасной вариант - ручной обход parquet-файлов
                from pathlib import Path

                # Ищем все файлы .parquet рекурсивно в директории данных
                parquet_files = list(Path(self.data_dir).rglob("*.parquet"))
                
                # Создаем пустой список для хранения данных
                dfs = []

                total_files = len(parquet_files)
                print(f"Found {total_files} parquet files to process manually")

                if self.max_shards is not None:
                    parquet_files = parquet_files[: self.max_shards]

                for file_path in parquet_files:
                    path = Path(file_path)
                    
                    # Извлекаем информацию о символе из пути
                    symbol_dir = path.parent.parent.parent.name
                    symbol = symbol_dir.replace("symbol=", "")
                    
                    # Читаем только необходимые колонки
                    df = pd.read_parquet(file_path, columns=["timestamp", "close"])
                    
                    # Добавляем информацию о символе
                    df["symbol"] = symbol
                    
                    # Проверяем и преобразуем timestamp в корректный формат datetime
                    if not df.empty:
                        # Если timestamp - это число, преобразуем его
                        if isinstance(df['timestamp'].iloc[0], (int, float, np.integer, np.floating)):
                            # Если это unix timestamp в миллисекундах (больше 2001 года в миллисекундах)
                            if df['timestamp'].iloc[0] > 1e12:  
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                if path.parent.parent.parent.name == "symbol=BTCUSDT" and path.parent.parent.name == "year=2022" and path.parent.name == "month=01":
                                    print(f"PANDAS DEBUG: Преобразован timestamp из миллисекунд: {df['timestamp'].head(3)}")
                            else:
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        
                        # Если timestamp уже в формате datetime, убедимся что timezone удален
                        elif isinstance(df['timestamp'].iloc[0], pd.Timestamp) and df['timestamp'].dt.tz is not None:
                            logger.debug(f"Удаляем timezone {df['timestamp'].dt.tz} из timestamp при ручной загрузке")
                            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                    
                    # Фильтруем по дате используя строгое сравнение datetime
                    if not df.empty:
                        print(f"DEBUG Filter: {path}, start_date={start_date}, end_date={end_date}")
                        print(f"DEBUG Filter: timestamp type={type(df['timestamp'].iloc[0])}, first timestamp={df['timestamp'].iloc[0]}")
                        # Убедимся, что start_date и end_date без timezone
                        start_date_naive = start_date.tz_localize(None) if hasattr(start_date, 'tz_localize') and start_date.tzinfo is not None else start_date
                        end_date_naive = end_date.tz_localize(None) if hasattr(end_date, 'tz_localize') and end_date.tzinfo is not None else end_date
                        
                        # Убедимся, что timestamp в df без timezone
                        if not df.empty and hasattr(df['timestamp'].iloc[0], 'tzinfo') and df['timestamp'].iloc[0].tzinfo is not None:
                            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                            
                        # Теперь сравнение будет корректным (без timezone с обеих сторон)
                        mask = (df["timestamp"] >= start_date_naive) & (df["timestamp"] <= end_date_naive)
                        filtered_df = df.loc[mask]
                        if not filtered_df.empty and path.parent.parent.parent.name == "symbol=BTCUSDT" and path.parent.parent.name == "year=2022" and path.parent.name == "month=01":
                            print(f"DEBUG Filter: Найдено {len(filtered_df)} строк после фильтрации по дате")
                    else:
                        filtered_df = df
                    
                    # Добавляем в список, если фрейм не пуст
                    if not filtered_df.empty:
                        dfs.append(filtered_df)
                
                # Если нет данных, возвращаем пустой фрейм
                if not dfs:
                    print("No data found after manual loading")
                    return pd.DataFrame()
                
                # Объединяем все данные
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Преобразуем в широкий формат
                wide_df = combined_df.pivot_table(
                    index="timestamp",
                    columns="symbol",
                    values="close",
                    aggfunc="last",
                )
                wide_df = wide_df.sort_index()

                self._freq = pd.infer_freq(wide_df.index)
                if self._freq:
                    wide_df = wide_df.asfreq(self._freq)

                print(f"Successfully loaded data manually. Shape: {wide_df.shape}")
                return wide_df
                
            except Exception as e2:
                # Если оба подхода не сработали, логируем ошибку
                print(f"Error loading data manually: {str(e2)}")
                return pd.DataFrame()