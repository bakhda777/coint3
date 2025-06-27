# Architecture

```mermaid
graph TD
    subgraph "1. Scanning"
        A[CLI: scan] --> B{Orchestrator}
        B --> C[Get All Symbols]
        C --> D[Create Pair List]
        D --> E{Dask Delayed Tasks}
        E --> F[Load Pair Data (Lazy)]
        F --> G[Coint Test]
    end
    subgraph "2. Backtesting"
        H[CLI: backtest] --> I{DataHandler}
        I --> J[Load Pair Data]
        J --> K[PairBacktester]
        K --> L[Calculate Metrics]
    end
    M[CLI: run-pipeline] --> B
    G --> M
    M --> I
```
