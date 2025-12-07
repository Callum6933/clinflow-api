from clinflow.config import load_config
from clinflow.data.load import load_dataset
from clinflow.models.train import train_model
from pathlib import Path
import copy
import pytest
import random

# get cfg
cfg = load_config()

# load clean dataset
clean_data_path = (
    Path(cfg["paths"]["processed_data"]["folder"])
    / cfg["paths"]["processed_data"]["file"]
)
df = load_dataset(clean_data_path)

def test_returns_expected_values():
    model = train_model(df, cfg)

    if "pipeline" not in model:
        raise ValueError("Model dict does not contain ('pipeline')")
    if "accuracy_score" not in model:
        raise ValueError("Model dict does not contain ('accuracy_score')")
    if "roc_auc_score" not in model:
        raise ValueError("Model dict does not contain ('roc_auc_score')")


def test_change_config():
    # train with full features
    model_full = train_model(df, cfg)
    full_pipeline = model_full["pipeline"]

    # train with one less numerical feature
    test_cfg = copy.deepcopy(cfg)
    test_cfg["model_training"]["numerical_features"].pop()
    model_reduced = train_model(df, test_cfg)
    reduced_pipeline = model_reduced["pipeline"]

    # verify that preprocessors are different
    full_transformer = full_pipeline.named_steps['preprocessor']
    reduced_transformer = reduced_pipeline.named_steps['preprocessor']
    assert full_transformer.transformers_[0][2] != reduced_transformer.transformers_[0][2]

def test_break_config():
    test_cfg = copy.deepcopy(cfg)
    features = test_cfg["model_training"]["numerical_features"]

    random_idx = random.randrange(len(features))

    bad_column_name = "INVALID_COLUMN_TEST"

    features[random_idx] = bad_column_name

    with pytest.raises((KeyError, ValueError)):
        train_model(df, test_cfg)
    

