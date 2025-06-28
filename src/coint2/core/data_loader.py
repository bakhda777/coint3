import dask.dataframe as dd
import pandas as pd  # type: ignore
from pathlib import Path
from typing import List
import time
import numpy as np

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
        """Load the full dataset using Dask.
        
        Returns:
        -------
        dask.dataframe.DataFrame
            Lazy Dask DataFrame with the data.
        """
        if self._all_data_cache is not None:
            return self._all_data_cache

        if not self.data_dir.exists():
            self._all_data_cache = dd.from_pandas(pd.DataFrame(), npartitions=1)
            return self._all_data_cache

        try:
            # Использование dd.read_parquet с правильными настройками
            ddf = dd.read_parquet(
                self.data_dir,
                engine="pyarrow",
                columns=["timestamp", "close"],
                ignore_metadata_file=True,  # Игнорируем _metadata файл
                calculate_divisions=False,   # Избегаем авто-расчёта разделений
                # В партициях используется ведущий ноль для месяцев (month=01, month=02 и т.д.)
                filters=None,  # Не применяем фильтры на уровне партиций
                validate_schema=False,  # Отключаем строгую валидацию схемы
            )
            return ddf
        except Exception as e:
            print(f"Ошибка загрузки данных через Dask в _load_full_dataset: {str(e)}")
            
            try:
                # Запасной вариант: использование glob для обхода всех файлов
                import glob
                import pandas as pd
                from pathlib import Path
                import dask.dataframe as dd
                
                print("Попытка загрузки через glob-обход файлов...")
                
                # Собираем все файлы data.parquet рекурсивно
                parquet_files = glob.glob(str(self.data_dir) + "/**/data.parquet", recursive=True)
                if not parquet_files:
                    print("Не найдены файлы data.parquet")
                    return dd.from_pandas(pd.DataFrame(), npartitions=1)
                    
                print(f"Найдено {len(parquet_files)} файлов data.parquet")
                
                # Создаем список Dask DataFrame из каждого файла
                dfs = []
                for file_path in parquet_files:
                    path = Path(file_path)
                    
                    # Определяем symbol из пути (3 уровня вверх от файла: symbol=XXX/year=YYYY/month=MM/data.parquet)
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
                        
                        # Диагностическая информация о формате timestamp (только для выбранного файла)
                        if not file_df.empty and path.parent.parent.parent.name == "symbol=BTCUSDT" and path.parent.parent.name == "year=2022" and path.parent.name == "month=01":
                            print(f"DEBUG: Файл {file_path}")
                            print(f"DEBUG: Тип timestamp: {type(file_df['timestamp'].iloc[0])}")
                            print(f"DEBUG: Первые timestamp: {file_df['timestamp'].head(3).tolist()}")
                        
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
                        
                        # Если timestamp уже в формате datetime, убедимся что timezone установлен
                        if not file_df.empty and isinstance(file_df['timestamp'].iloc[0], pd.Timestamp):
                            # Если нет timezone, устанавливаем UTC
                            if file_df['timestamp'].dt.tz is None:
                                file_df['timestamp'] = file_df['timestamp'].dt.tz_localize('UTC')
                        
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
                    return dd.from_pandas(pd.DataFrame(), npartitions=1)
                    
                try:
                    # Объединяем все Dask DataFrame в один
                    combined_ddf = dd.concat(dfs)
                    print(f"Успешно создан объединенный Dask DataFrame")
                    return combined_ddf
                except Exception as concat_error:
                    print(f"Ошибка при объединении DataFrame: {str(concat_error)}")
                    return dd.from_pandas(pd.DataFrame(), npartitions=1)
                    
            except Exception as e2:
                print(f"Ошибка при использовании запасного варианта: {str(e2)}")
                # Если и запасной вариант не сработал, возвращаем пустой фрейм
                return dd.from_pandas(pd.DataFrame(), npartitions=1)

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

        # Преобразуем в широкий формат с символами в столбцах
        wide_ddf = filtered_ddf.pivot_table(
            index="timestamp", columns="symbol", values="close"
        )

        # Ожидаем вычисления и преобразуем в pandas DataFrame
        wide_pdf = wide_ddf.compute()
        if wide_pdf.empty:
            return pd.DataFrame()

        # Сортируем по индексу (дате)
        wide_pdf = wide_pdf.sort_index()
        return wide_pdf

    def load_pair_data(
        self,
        symbol1: str,
        symbol2: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Load and align data for two symbols within the given date range."""
        ddf = self._load_full_dataset()

        # Проверка на пустой DataFrame
        if not ddf.columns.compute().tolist():
            return pd.DataFrame()
            
        # Фильтруем только интересующие нас символы
        pair_ddf = ddf[ddf["symbol"].isin([symbol1, symbol2])]

        # Конвертируем в pandas для дальнейшей обработки
        pair_pdf = pair_ddf.compute()

        # Проверка на пустой результат
        if pair_pdf.empty:
            return pd.DataFrame()

        # Убедимся, что timestamp в формате datetime
        pair_pdf["timestamp"] = pd.to_datetime(pair_pdf["timestamp"])
        
        # Фильтруем по диапазону дат
        mask = (pair_pdf["timestamp"] >= pd.Timestamp(start_date)) & (
            pair_pdf["timestamp"] <= pd.Timestamp(end_date)
        )
        pair_pdf = pair_pdf.loc[mask]

        # Проверка на пустой результат после фильтрации
        if pair_pdf.empty:
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
        freq = pd.infer_freq(wide_df.index) or "D"
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
        start_time = time.time()
        data_df = self.preload_all_data(start_date, end_date)
        elapsed_time = time.time() - start_time
        print(f"Данные загружены за {elapsed_time:.2f} секунд. Размер: {data_df.shape}")

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
        if not self.data_dir.exists():
            return pd.DataFrame()

        try:
            # Загружаем данные через Dask с оптимизированными параметрами
            ddf = self._load_full_dataset()

            # Проверка на пустой DataFrame
            if not ddf.columns.compute().tolist():
                print("No columns found in dataset")
                return pd.DataFrame()

            # Конвертируем timestamp в datetime
            ddf["timestamp"] = dd.to_datetime(ddf["timestamp"])
            
            # Фильтруем данные по заданному диапазону дат
            filtered_ddf = ddf[
                (ddf["timestamp"] >= pd.Timestamp(start_date)) & 
                (ddf["timestamp"] <= pd.Timestamp(end_date))
            ]

            # Преобразуем в широкий формат с символами в столбцах
            wide_ddf = filtered_ddf.pivot_table(
                index="timestamp", columns="symbol", values="close"
            )

            # Вычисляем конечный результат
            wide_pdf = wide_ddf.compute()
            
            if wide_pdf.empty:
                print(f"No data found between {start_date} and {end_date}")
                return pd.DataFrame()

            # Сортируем по индексу (датам)
            return wide_pdf.sort_index()
            
        except Exception as e:
            # Если Dask-подход не сработал, переходим на ручной обход через pandas
            print(f"Error loading data via Dask: {str(e)}")
            
            try:
                # Запасной вариант - ручной обход parquet-файлов
                import glob
                from pathlib import Path
                
                # Ищем все файлы .parquet рекурсивно в директории данных
                parquet_files = glob.glob(str(self.data_dir) + "/**/data.parquet", recursive=True)
                
                # Создаем пустой список для хранения данных
                dfs = []
                
                total_files = len(parquet_files)
                print(f"Found {total_files} parquet files to process manually")
                
                # Ограничиваем количество файлов для быстрой загрузки
                file_limit = min(total_files, 500)
                
                for file_path in parquet_files[:file_limit]:  
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
                        
                        # Если timestamp уже в формате datetime, убедимся что timezone установлен
                        elif isinstance(df['timestamp'].iloc[0], pd.Timestamp) and df['timestamp'].dt.tz is None:
                            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                    
                    # Фильтруем по дате используя строгое сравнение datetime
                    if not df.empty:
                        print(f"DEBUG Filter: {path}, start_date={start_date}, end_date={end_date}")
                        print(f"DEBUG Filter: timestamp type={type(df['timestamp'].iloc[0])}, first timestamp={df['timestamp'].iloc[0]}")
                        mask = (df["timestamp"] >= pd.Timestamp(start_date)) & (df["timestamp"] <= pd.Timestamp(end_date))
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
                wide_df = combined_df.pivot_table(index="timestamp", columns="symbol", values="close")
                wide_df = wide_df.sort_index()
                
                print(f"Successfully loaded data manually. Shape: {wide_df.shape}")
                return wide_df
                
            except Exception as e2:
                # Если оба подхода не сработали, логируем ошибку
                print(f"Error loading data manually: {str(e2)}")
                return pd.DataFrame()