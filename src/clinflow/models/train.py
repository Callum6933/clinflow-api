from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from clinflow.config import load_config
from clinflow.logging_utils import get_logger
from clinflow.data.load import load_dataset
from pathlib import Path
import json
import numpy as np


def train_model(df, cfg):
    # setup configuration parameters
    target_column_name = cfg["model_training"]["target_column_name"]
    drop_column_name = cfg["model_training"]["exclude_columns"]
    test_size = cfg["model_training"]["test_size"]
    random_state = cfg["model_training"]["random_state"]
    numerical_columns = cfg["model_training"]["numerical_features"]
    categorical_columns = cfg["model_training"]["categorical_features"]
    model_params = cfg["model_training"]["model_params"]

    # configure logging
    logger = get_logger(__name__)

    # create X (features) and y (target)
    y = df[target_column_name]
    X = df.drop(drop_column_name, axis=1)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Test split created")

    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(steps=[("scaler", OneHotEncoder())])

    # configure specific transformers for numerical and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**model_params)),
        ]
    )
    logger.info("Pipeline created")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    logger.info("Accuracy scores calculated")

    # create dict with model, scaler and metrics
    model = {
        "pipeline": pipeline,
        "y_test": y_test,
        "y_pred": y_pred,
    }

    return model


def evaluate_model(y_test, y_pred):
    # compute_metrics
    metrics = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # create results dir if not already exists
    cfg = load_config()
    Path(cfg['model_training']['path_to_results']['directory']).mkdir(exist_ok=True)

    # write to json file
    file_path = Path(f"{cfg['model_training']['path_to_results']['directory']}{cfg['model_training']['path_to_results']['file']}")
    with open(file_path, "w") as file:
        json.dump(metrics, file, indent=2, default=float)

    return file_path

def main():
    # configure logger
    logger = get_logger(__name__)

    # get cfg
    cfg = load_config()

    # load clean dataset
    clean_data_path = (
        Path(cfg["paths"]["processed_data"]["folder"])
        / cfg["paths"]["processed_data"]["file"]
    )
    df = load_dataset(clean_data_path)

    # train model
    model = train_model(df, cfg)
    logger.info("Model training successful")

    # evaluate model
    y_test = model["y_test"]
    y_pred = model["y_pred"]

    try:
        path = evaluate_model(y_test, y_pred)
    except Exception as e:
        logger.error(f"Model evaluation error: {e}")
        raise
    
    logger.info(f"Model evaluation successful. File saved to '{path}'")



if __name__ == "__main__":
    main()
