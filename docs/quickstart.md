# Quickstart

This guide walks you through preparing data, configuring the project and running the pipeline.

## 1. Prepare data

Price data should be stored as partitioned parquet files inside the directory specified by `data_dir` (by default `data_optimized`). Each symbol lives in its own partition:

```
data_optimized/
  symbol=BTCUSDT/
    year=2021/
      month=01/
        data.parquet
```

The parquet files must contain the columns `timestamp`, `close` and `symbol`.

## 2. Configure

Edit `configs/main.yaml` to match your dataset location and desired backtest parameters. Important options:

```yaml
data_dir: "data_optimized"
results_dir: "results"
pair_selection:
  lookback_days: 90
  coint_pvalue_threshold: 0.05
backtest:
  timeframe: "1d"
  rolling_window: 30
  zscore_threshold: 1.5
  fill_limit_pct: 0.2  # доля подряд идущих пропусков, заполняемых при ffill/bfill
max_shards: null
```

## 3. Run the project

Install dependencies using `poetry install` then execute commands using `poetry run`.

```bash
# Find all cointegrated pairs
poetry run coint2 scan

# Backtest a specific pair
poetry run coint2 backtest --pair BTCUSDT,ETHUSDT

# Run scanning and backtesting sequentially
poetry run coint2 run-pipeline
```

Metrics for each backtest are written to the directory defined by `results_dir`.
