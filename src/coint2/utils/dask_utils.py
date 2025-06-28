import pandas as pd
import dask.dataframe as dd


def empty_ddf() -> dd.DataFrame:
    """Return an empty Dask DataFrame."""
    return dd.from_pandas(pd.DataFrame(), npartitions=1)
