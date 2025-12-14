from clinflow.models.io import save_model, load_model
from clinflow.models.train import train_model
from clinflow.config import load_config
from clinflow.logging_utils import get_logger
from clinflow.data.load import load_dataset
from pathlib import Path
import numpy as np
import pytest

logger = get_logger(__name__)

cfg = load_config()

clean_data_path = Path(cfg["paths"]["processed_data"]["folder"]) / cfg["paths"]["processed_data"]["file"]
df = load_dataset(clean_data_path)

# train model
try:
    model = train_model(df, cfg)
except Exception as e:
    logger.error(f"Error while training model: {e}")
    raise

def test_save_model():
    save_model(model)       # no filepath
    save_model(model, "models/model.joblib")       # with filepath

def test_load_model():
    load_model()                             # without filepath
    load_model("models/model.joblib")        # with filepath

def test_compare_predictions():
    # BEFORE/AFTER COMPARISON
    # equality is assessed as whether the predictions are the same    
    # save/reload model
    before_model = model["pipeline"]
    path = save_model(model)
    after_model = load_model(path)

    # drop target and num columns
    X_test = df.drop(["target", "num"], axis=1)

    before_pred = before_model.predict(X_test)
    after_pred = after_model.predict(X_test)

    assert np.array_equal(before_pred, after_pred)

