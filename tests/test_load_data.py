import pytest
from clinflow.data.load import load_raw_data
import pandas as pd

def test_returns_dataframe_with_args():
    # no CLI args
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)

    # with CLI args
    df = load_raw_data(filepath="data/raw/heart_disease_dataset.csv")
    assert isinstance(df, pd.DataFrame)


def test_num_exists():
    df = load_raw_data("data/raw/heart_disease_dataset.csv")
    assert "num" in df.columns

def test_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_raw_data("data/foo/bar.csv")
