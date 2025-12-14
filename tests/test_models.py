import pytest
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from clinflow.models.train import train_model, evaluate_model
from clinflow.pipeline import run_data_pipeline
from clinflow.config import load_config
from clinflow.data.load import load_dataset
from clinflow.logging_utils import get_logger
import copy
import json

cfg = load_config()
logger = get_logger(__name__)

def test_accuracy_beats_baseline():
    # train model
    clean_data_path = (
        Path(cfg["paths"]["processed_data"]["folder"])
        / cfg["paths"]["processed_data"]["file"]
    )
    df = load_dataset(clean_data_path)
    
    model = train_model(df, cfg)
    y_test = model["y_test"]
    y_pred = model["y_pred"]

    # train dummy classifier
    target_column_name = cfg["model_training"]["target_column_name"]
    drop_column_name = cfg["model_training"]["exclude_columns"]
    test_size = cfg["model_training"]["test_size"]
    random_state = cfg["model_training"]["random_state"]

    y = df[target_column_name]
    X = df.drop(drop_column_name, axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_accuracy = dummy.score(X_test, y_test)

    # test accuracy beats baseline
    path_to_metrics = evaluate_model(y_test, y_pred)
    with open(path_to_metrics, "r") as f:
        metrics = json.load(f)

    real_accuracy = metrics["accuracy"]

    assert real_accuracy > dummy_accuracy

def test_metrics_in_valid_ranges():
    # train model
    clean_data_path = (
        Path(cfg["paths"]["processed_data"]["folder"])
        / cfg["paths"]["processed_data"]["file"]
    )
    df = load_dataset(clean_data_path)
    
    model = train_model(df, cfg)
    y_test = model["y_test"]
    y_pred = model["y_pred"]


    # load metrics.json
    path_to_metrics = evaluate_model(y_test, y_pred)
    with open(path_to_metrics, "r") as f:
        metrics = json.load(f)

    # accuracy should be between 0 and 1
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # check class "0" metrics
    assert 0.0 <= metrics["0"]["precision"] <= 1.0
    assert 0.0 <= metrics["0"]["recall"] <= 1.0
    assert 0.0 <= metrics["0"]["f1-score"] <= 1.0
    assert metrics["0"]["support"] > 0

    # check class "1" metrics
    assert 0.0 <= metrics["1"]["precision"] <= 1.0
    assert 0.0 <= metrics["1"]["recall"] <= 1.0
    assert 0.0 <= metrics["1"]["f1-score"] <= 1.0
    assert metrics["1"]["support"] > 0

    # check macro avg metrics
    assert 0.0 <= metrics["macro avg"]["precision"] <= 1.0
    assert 0.0 <= metrics["macro avg"]["recall"] <= 1.0
    assert 0.0 <= metrics["macro avg"]["f1-score"] <= 1.0
    assert metrics["macro avg"]["support"] > 0

    # check weighted avg metrics
    assert 0.0 <= metrics["weighted avg"]["precision"] <= 1.0
    assert 0.0 <= metrics["weighted avg"]["recall"] <= 1.0
    assert 0.0 <= metrics["weighted avg"]["f1-score"] <= 1.0
    assert metrics["weighted avg"]["support"] > 0

def test_model_artifacts_exist():
    # run training pipeline
    run_data_pipeline()

    # asssert metrics exist
    assert Path(f"{cfg['model_training']['path_to_results']['directory']}{cfg['model_training']['path_to_results']['file']}").exists()
    # assert model exists
    assert Path(f"{cfg['model_training']['path_to_model']['directory']}{cfg['model_training']['path_to_model']['file']}").exists()

@pytest.mark.parametrize("test_size", [0.2, 0.3])
@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_different_configs(test_size, C):
    # model config with test_size and C
    test_cfg = copy.deepcopy(cfg)
    test_cfg["model_training"]["test_size"] = test_size
    test_cfg["model_training"]["model_params"]["C"] = C

    # train model
    clean_data_path = (
        Path(cfg["paths"]["processed_data"]["folder"])
        / cfg["paths"]["processed_data"]["file"]
    )
    df = load_dataset(clean_data_path)
    
    train_model(df, cfg)
    logger.info(f"Model training successful with params: test size:'{test_size}', C: '{C}'")




    pass