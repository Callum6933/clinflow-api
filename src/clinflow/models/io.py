import joblib
from pathlib import Path
from clinflow.config import load_config
from clinflow.logging_utils import get_logger


def save_model(model, filepath=None):
    """Save a trained machine learning model pipeline to disk using joblib.

    Serializes the model pipeline to a file for later use in production or evaluation.
    Creates the parent directory if it doesn't exist. Uses joblib for efficient
    serialization of scikit-learn pipelines and transformers.

    Args:
        model (dict): Dictionary containing the trained model with key "pipeline".
            The "pipeline" value should be a scikit-learn Pipeline object or
            compatible model object that can be serialized with joblib.
        filepath (str or Path, optional): Path where the model should be saved.
            If None, uses the default path from configuration:
            cfg['model_training']['path_to_model']['directory']/['file'].
            Defaults to None.

    Returns:
        Path: The filepath where the model was saved.

    Raises:
        Exception: If joblib.dump() fails (e.g., permissions error, disk full).
            The original exception is logged and re-raised.

    Side Effects:
        - Creates parent directory if it doesn't exist
        - Writes serialized model to disk
        - Logs filepath and save success/failure

    Examples:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.linear_model import LogisticRegression
        >>> pipeline = Pipeline([("clf", LogisticRegression())])
        >>> model = {"pipeline": pipeline}
        >>>
        >>> # Save with default path from config
        >>> path = save_model(model)
        >>> print(path)
        models/heart_disease_model.pkl
        >>>
        >>> # Save to custom location
        >>> path = save_model(model, filepath="exports/my_model.pkl")

    Note:
        Only the "pipeline" key from the model dict is saved. Other keys
        (like metrics or metadata) are not persisted. Consider saving those
        separately if needed for model tracking.
    """
    cfg = load_config()
    logger = get_logger(__name__)

    # create parent directory if not exists
    if filepath is None:
        filepath = (
            Path(cfg["model_training"]["path_to_model"]["directory"])
            / cfg["model_training"]["path_to_model"]["file"]
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
    """Load a trained machine learning model pipeline from disk using joblib.

    Deserializes a previously saved model pipeline for inference, evaluation, or
    further training. The loaded model can be used immediately for predictions.

    Args:
        filepath (str or Path, optional): Path to the saved model file.
            If None, uses the default path from configuration:
            cfg['model_training']['path_to_model']['directory']/['file'].
            Defaults to None.

    Returns:
        Pipeline: The loaded scikit-learn Pipeline object or compatible model.
            Ready to use for predictions via .predict() or .predict_proba().

    Raises:
        FileNotFoundError: If the specified filepath doesn't exist.
        Exception: If joblib.load() fails (e.g., corrupted file, version mismatch).
            The original exception is logged and re-raised.

    Side Effects:
        Logs the filepath being loaded and success/failure status.

    Examples:
        >>> # Load model from default config path
        >>> pipeline = load_model()
        >>> predictions = pipeline.predict(X_test)
        >>>
        >>> # Load from custom location
        >>> pipeline = load_model(filepath="exports/my_model.pkl")
        >>> proba = pipeline.predict_proba(X_new)
        >>> print(proba)
        [[0.23 0.77]
         [0.81 0.19]]

    Note:
        - Ensure the scikit-learn version used to save the model is compatible
          with the version used to load it. Major version mismatches may cause errors.
        - The loaded object is the pipeline only; if you saved metrics separately,
          load those from their own files.
        - For production use, validate the model after loading (check expected
          attributes, feature names, etc.) before making predictions.
    """
    cfg = load_config()
    logger = get_logger(__name__)

    if filepath is None:
        filepath = (
            Path(cfg["model_training"]["path_to_model"]["directory"])
            / cfg["model_training"]["path_to_model"]["file"]
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
