import dask.dataframe as dd

from coint2.utils import empty_ddf


def test_empty_ddf_returns_empty_dataframe() -> None:
    ddf = empty_ddf()
    assert isinstance(ddf, dd.DataFrame)
    assert ddf.compute().empty
