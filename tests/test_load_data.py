import pytest
from clinflow.data.load import load_dataset
from clinflow.config import load_config
import pandas as pd
from pathlib import Path

def test_returns_dataframe_with_args():
    # no CLI args
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)

    # with CLI args
    cfg = load_config()
    filepath = Path(cfg["paths"]["raw_data"]["folder"]) / cfg["paths"]["raw_data"]["file"]

    df = load_dataset(filepath)
    assert isinstance(df, pd.DataFrame)


def test_num_exists():
    df = load_dataset("data/raw/heart_disease_dataset.csv")
    assert "num" in df.columns

def test_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_dataset("data/foo/bar.csv")
