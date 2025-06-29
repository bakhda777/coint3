#!/usr/bin/env python3
"""
Скрипт для проверки и удаления дубликатов по timestamp в parquet-файлах.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm


def check_and_remove_duplicates(file_path: str, save_cleaned: bool = True) -> tuple:
    """
    Проверяет наличие дубликатов timestamp в parquet-файле и опционально создает очищенную версию.
    
    Args:
        file_path: Путь к parquet-файлу
        save_cleaned: Флаг, указывающий нужно ли сохранять очищенный файл
        
    Returns:
        tuple: (общее кол-во строк, кол-во дубликатов, кол-во уникальных строк)
    """
    try:
        # Загрузка данных с pyarrow
        df = pd.read_parquet(file_path)
        total_rows = len(df)
        
        # Преобразование timestamp в datetime если необходимо
        if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], (int, float, np.integer, np.floating)):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
        # Подсчет дубликатов по timestamp
        duplicates = df.duplicated(subset=['timestamp'], keep='first')
        duplicates_count = duplicates.sum()
        unique_rows = total_rows - duplicates_count
        
        if save_cleaned and duplicates_count > 0:
            # Удаление дубликатов, сохраняя только первое вхождение для каждого timestamp
            df_unique = df.drop_duplicates(subset=['timestamp'], keep='first')
            
            # Создаем путь для сохранения очищенного файла
            file_path_obj = Path(file_path)
            dir_path = file_path_obj.parent
            clean_path = dir_path / "data_clean.parquet"
            
            # Сохраняем файл без дубликатов
            df_unique.to_parquet(clean_path, index=False)
            
            # Если нужно заменить исходный файл, создаем бэкап и затем заменяем
            if args.replace:
                backup_path = dir_path / "data_backup.parquet"
                shutil.copy2(file_path, backup_path)
                shutil.move(str(clean_path), file_path)
                
        return total_rows, duplicates_count, unique_rows
        
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return 0, 0, 0


def process_directory(data_dir: str, replace: bool = False, limit: int = None) -> None:
    """
    Обрабатывает все parquet-файлы в указанной директории.
    
    Args:
        data_dir: Путь к директории с данными
        replace: Флаг, указывающий нужно ли заменять исходные файлы очищенными
        limit: Ограничение на количество обрабатываемых файлов (None = все файлы)
    """
    # Проверка существования директории
    if not os.path.exists(data_dir):
        print(f"Директория {data_dir} не существует.")
        return
    
    print(f"Анализ parquet-файлов в директории {data_dir}")
    
    # Поиск parquet-файлов
    parquet_files = glob.glob(os.path.join(data_dir, "**/data.parquet"), recursive=True)
    total_files = len(parquet_files)
    print(f"Найдено {total_files} файлов.")
    
    if limit and limit < total_files:
        parquet_files = parquet_files[:limit]
        print(f"Ограничено до {limit} файлов.")
    
    total_duplicates = 0
    total_rows = 0
    total_unique = 0
    files_with_duplicates = 0
    
    print("\nНачинаем обработку файлов...")
    for file_path in tqdm(parquet_files, desc="Обработка файлов"):
        rows, duplicates, unique = check_and_remove_duplicates(file_path, save_cleaned=True)
        
        total_rows += rows
        total_duplicates += duplicates
        total_unique += unique
        
        if duplicates > 0:
            files_with_duplicates += 1
            duplicate_pct = (duplicates / rows) * 100 if rows > 0 else 0
            print(f"\n{file_path}: {duplicates} дубликатов из {rows} строк ({duplicate_pct:.2f}%)")
    
    # Итоговая статистика
    print("\n=== Итоговая статистика ===")
    print(f"Проверено файлов: {len(parquet_files)}")
    print(f"Найдено файлов с дубликатами: {files_with_duplicates} ({(files_with_duplicates/len(parquet_files))*100:.2f}%)")
    print(f"Общее количество строк в файлах: {total_rows}")
    print(f"Общее количество дубликатов: {total_duplicates} ({(total_duplicates/total_rows)*100:.2f}% от всех строк)")
    print(f"Общее количество уникальных строк: {total_unique}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка и удаление дубликатов в parquet-файлах")
    parser.add_argument("--data_dir", type=str, default="./data_optimized", 
                        help="Директория с parquet-файлами")
    parser.add_argument("--replace", action="store_true", 
                        help="Заменить исходные файлы очищенными версиями")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Ограничение на количество обрабатываемых файлов")
    args = parser.parse_args()
    
    process_directory(args.data_dir, args.replace, args.limit)
