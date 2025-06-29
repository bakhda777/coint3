#!/usr/bin/env python3
"""
Скрипт для проверки структуры parquet-файлов и анализа 15-минутных интервалов.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def inspect_parquet_file(file_path):
    """Проверяет структуру parquet-файла и выводит информацию о временных метках."""
    print(f"\n=== Анализ файла {file_path} ===")
    
    try:
        # Загрузка данных с pyarrow
        df = pd.read_parquet(file_path)
        
        print(f"Размер DataFrame: {df.shape}")
        print(f"Столбцы: {df.columns.tolist()}")
        
        # Проверка наличия timestamp
        if 'timestamp' not in df.columns:
            print("ОШИБКА: Столбец timestamp отсутствует!")
            return

        # Информация о типе timestamp
        ts_type = type(df['timestamp'].iloc[0])
        print(f"Тип timestamp: {ts_type}")
        
        # Преобразование timestamp в datetime если необходимо
        if isinstance(df['timestamp'].iloc[0], (int, float, np.integer, np.floating)):
            print("Преобразование timestamp из числового формата в datetime...")
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
        # Вывод первых и последних временных меток
        print(f"Первые 5 timestamp: {df['timestamp'].head(5).tolist()}")
        print(f"Последние 5 timestamp: {df['timestamp'].tail(5).tolist()}")
        
        # Анализ 15-минутных интервалов
        # Получаем уникальные минуты
        unique_minutes = sorted(df['timestamp'].dt.minute.unique())
        print(f"Уникальные значения минут: {unique_minutes}")
        
        # Проверка на 15-минутные интервалы
        expected_minutes = [0, 15, 30, 45]
        is_15min = all(minute in expected_minutes for minute in unique_minutes) and all(expected in unique_minutes for expected in expected_minutes)
        print(f"Соответствует 15-минутным интервалам: {is_15min}")
        
        # Гистограмма распределения по часам и минутам
        print("Распределение по часам:")
        hour_counts = df['timestamp'].dt.hour.value_counts().sort_index()
        for hour, count in hour_counts.items():
            print(f"  Час {hour}: {count} записей")
            
        print("Распределение по минутам:")
        minute_counts = df['timestamp'].dt.minute.value_counts().sort_index()
        for minute, count in minute_counts.items():
            print(f"  Минута {minute}: {count} записей")
            
        # Проверка интервалов между последовательными метками
        df_sorted = df.sort_values('timestamp')
        df_sorted['diff'] = df_sorted['timestamp'].diff().dt.total_seconds() / 60
        
        # Удаляем первую запись с NaN разницей
        diff_stats = df_sorted['diff'].dropna()
        
        if len(diff_stats) > 0:
            print("Статистика интервалов (в минутах):")
            print(f"  Минимальный интервал: {diff_stats.min()}")
            print(f"  Максимальный интервал: {diff_stats.max()}")
            print(f"  Средний интервал: {diff_stats.mean()}")
            print(f"  Медианный интервал: {diff_stats.median()}")
            
            # Проверка, соответствуют ли интервалы 15 минутам
            counts_15min = (diff_stats == 15).sum()
            percent_15min = (counts_15min / len(diff_stats)) * 100
            print(f"  Интервал точно 15 минут: {counts_15min} записей ({percent_15min:.2f}%)")
            
            # Основные интервалы 
            top_intervals = diff_stats.value_counts().head(5)
            print("  Самые частые интервалы:")
            for interval, count in top_intervals.items():
                print(f"    {interval} минут: {count} записей ({(count/len(diff_stats))*100:.2f}%)")
    
    except Exception as e:
        print(f"Ошибка при анализе файла: {e}")


def main():
    """Основная функция для анализа parquet-файлов."""
    # Базовая директория с данными
    data_dir = Path("./data_optimized")
    
    # Проверка существования директории
    if not data_dir.exists():
        print(f"Директория {data_dir} не существует.")
        return
    
    print(f"Анализ parquet-файлов в директории {data_dir}")
    
    # Поиск parquet-файлов
    parquet_files = list(glob.glob(str(data_dir) + "/**/data.parquet", recursive=True))
    print(f"Найдено {len(parquet_files)} файлов.")
    
    # Выбираем максимум 5 файлов для анализа (чтобы не перегружать вывод)
    symbols_to_check = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'STETHEUR']
    
    analyzed_files = 0
    for symbol in symbols_to_check:
        symbol_files = [f for f in parquet_files if f'symbol={symbol}/' in f]
        
        # Анализируем только первый найденный файл для каждого символа и года 2022
        for file_path in symbol_files:
            if '/year=2022/' in file_path or '/year=2023/' in file_path:
                inspect_parquet_file(file_path)
                analyzed_files += 1
                break
        
        if analyzed_files >= 5:
            break
    
    print("\nАнализ завершен.")


if __name__ == "__main__":
    main()
