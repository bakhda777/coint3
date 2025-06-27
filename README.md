# Coint2

Coint2 is a framework for researching cointegration based trading strategies. It provides tools to scan historical price data for cointegrated pairs and to backtest simple mean reversion systems.

See [docs/quickstart.md](docs/quickstart.md) for a hands-on guide.

## Features

- **Partitioned datasets** – price data is stored in `data_optimized/symbol=...` partitions for efficient access.
- **Parallel scanning** – uses Dask to test all symbol pairs concurrently.
- **Flexible configuration** – parameters are defined in `configs/main.yaml` and can be adjusted for your data.
- **Command line interface** – scan, backtest and run the full pipeline directly from the CLI.

## Installation

```bash
git clone https://github.com/yourname/coint2.git
cd coint2
poetry install
```

## Usage

Examples of running the main commands:

```bash
# Search for cointegrated pairs
poetry run coint2 scan

# Backtest a specific pair
poetry run coint2 backtest --pair BTCUSDT,ETHUSDT

# Execute scanning and backtesting in one step
poetry run coint2 run-pipeline
```
