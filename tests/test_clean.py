from clinflow.data.clean import validate_data
import pandas as pd
import pytest
import yaml

# load cfg
with open("config/config.yml", "r") as f:
    cfg = yaml.safe_load(f)

def test_valid_data():
    # load dataset and configuration
    df = pd.DataFrame({
        'age': range(50, 100),  # valid
        'sex': [0, 1] * 25,
        'target': [0, 1] * 25
    })

    validate_data(df, cfg)

def test_missing_values_fails():
    df = pd.DataFrame({
        "age": [50, 60] * 25, 
        "sex": [1, None] + list(range(51, 99)),    # None shouldn't pass
        "target": [0, 1] * 25,
    })
        
    with pytest.raises(ValueError, match="missing values"):
        validate_data(df, cfg)

def test_numerical_columns_fails():
    # create invalid dataset:
    df = pd.DataFrame({
        "age": ["50", "60"] * 25,   # numerical columns should be numerical
        "sex": [1, 0] * 25,
        "target": [0, 1] * 25,
    })

    with pytest.raises(ValueError, match="Numerical column"):
        validate_data(df, cfg)

def test_categorical_columns_fails():
    # create invalid dataset:
    df = pd.DataFrame({
        "age": [50, 60] * 25,
        "sex": ["male", "female"] * 25,     # categorical columns should be numerical
        "target": [0, 1] * 25,
    })

    with pytest.raises(ValueError, match="Categorical column"):
        validate_data(df, cfg)


def test_target_columns_not_exist_fails():
    # create invalid dataset:
    df = pd.DataFrame({
        "age": [50, 60] * 25,
        "sex": [1, 0] * 25,
        "num": [0, 1] * 25,     # target column doesn't exist
    })

    with pytest.raises(ValueError, match="does not exist"):
        validate_data(df, cfg)


def test_target_column_not_binary_fails():
    # create invalid dataset:
    df = pd.DataFrame({
        "age": [50, 60] * 25,
        "sex": [1, 0] * 25,
        "target": [2, 3] * 25,      # should only be 0/1
    })

    with pytest.raises(ValueError, match="target"):
        validate_data(df, cfg)

def test_unreasonable_ranges_fails():
    # create invalid dataset:
    df = pd.DataFrame({
        "age": [50, 130] * 25,      # 130 is outisde of range
        "sex": [1, 0] * 25,
        "target": [0, 1] * 25,
    })

    with pytest.raises(ValueError, match="outside reasonable range"):
        validate_data(df, cfg)


def test_not_enough_rows_fails():
    # create invalid dataset:
    df = pd.DataFrame({
        "age": [50, 60],    # only 2 rows (needs 50+)
        "sex": [1, 0],
        "target": [0, 1],
    })
    
    with pytest.raises(ValueError, match="Less than"):
        validate_data(df, cfg)


