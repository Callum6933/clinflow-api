import joblib
from pathlib import Path
from clinflow.config import load_config
from clinflow.logging_utils import get_logger


def save_model(model, filepath=None):
    cfg = load_config()
    logger = get_logger(__name__)

    # create parent directory if not exists
    if filepath == None:
        filepath = Path(
            f"{cfg['model_training']['path_to_model']['directory']}{cfg['model_training']['path_to_model']['file']}"
        )
        logger.info(f"Filepath not provided: using '{filepath}'")

    Path(filepath).parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"Directory '{filepath}' created successfully")

    # save model with joblib
    pipeline = model["pipeline"]

    try:
        joblib.dump(pipeline, filepath)
    except Exception as e:
        logger.error(f"Failed to save model to '{filepath}': {e}")
        raise

    logger.info(f"Model saved to '{filepath}'")

    return filepath


def load_model(filepath=None):
    cfg = load_config()
    logger = get_logger(__name__)

    if filepath == None:
        filepath = Path(
            f"{cfg['model_training']['path_to_model']['directory']}{cfg['model_training']['path_to_model']['file']}"
        )
        logger.info(f"Filepath not provided: using '{filepath}'")

    # load model with joblib
    try:
        pipeline = joblib.load(filepath)
    except Exception as e:
        logger.error(f"Failed to load file at '{filepath}': {e}")
        raise

    logger.info(f"Successfully loaded model from '{filepath}'")

    return pipeline
