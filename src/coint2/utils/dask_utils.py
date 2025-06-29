import dask.dataframe as dd
import pandas as pd


def empty_ddf() -> dd.DataFrame:
    """Return an empty Dask DataFrame."""
    return dd.from_pandas(pd.DataFrame(), npartitions=1)
