#!/usr/bin/env python3
"""
Скрипт для отображения первых 100 строк parquet файла.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def display_parquet_data(file_path: str, rows: int = 100) -> None:
    """
    Отображает первые N строк parquet файла.
    
    Args:
        file_path: Путь к parquet-файлу
        rows: Количество строк для отображения
    """
    try:
        # Загрузка данных
        df = pd.read_parquet(file_path)
        
        # Преобразование timestamp в datetime если необходимо
        if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], (int, float, np.integer, np.floating)):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Вывод информации о файле
        print(f"Файл: {file_path}")
        print(f"Общее количество строк: {len(df)}")
        print(f"Столбцы: {df.columns.tolist()}")
        print(f"Типы данных:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        # Проверка временных интервалов
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            time_diffs = df['timestamp'].diff().dropna()
            
            if len(time_diffs) > 0:
                # Расчет интервалов в минутах
                minutes_diffs = time_diffs.dt.total_seconds() / 60
                
                print("\nАнализ временных интервалов:")
                print(f"  Минимальный интервал: {minutes_diffs.min():.2f} минут")
                print(f"  Максимальный интервал: {minutes_diffs.max():.2f} минут")
                print(f"  Средний интервал: {minutes_diffs.mean():.2f} минут")
                print(f"  Медианный интервал: {minutes_diffs.median():.2f} минут")
                
                # Распределение интервалов
                interval_counts = minutes_diffs.value_counts().sort_index()
                print("\nРаспределение интервалов:")
                for interval, count in interval_counts.items():
                    print(f"  {interval:.2f} минут: {count} раз ({count/len(minutes_diffs)*100:.2f}%)")
        
        # Вывод первых N строк
        print("\nПервые", rows, "строк:")
        print(df.head(rows).to_string())
        
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Отображение данных из parquet файла")
    parser.add_argument("--file", type=str, required=True, 
                        help="Путь к parquet-файлу")
    parser.add_argument("--rows", type=int, default=100, 
                        help="Количество строк для отображения")
    args = parser.parse_args()
    
    display_parquet_data(args.file, args.rows)
