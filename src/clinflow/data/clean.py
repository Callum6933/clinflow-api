from clinflow.data.load import load_dataset
from clinflow.logging_utils import get_logger
from clinflow.config import load_config
from pathlib import Path
import pandas as pd


def clean_data(df, cfg):
    """Clean and transform raw heart disease dataset for machine learning.

    Performs data cleaning operations including missing value handling, type
    conversions, and target variable encoding. The function applies configuration-
    driven transformations to prepare data for model training.

    Args:
        df (pd.DataFrame): Raw dataset to clean. Should contain heart disease
            patient data with columns specified in the configuration.
        cfg (dict): Configuration dictionary containing cleaning parameters:
            - "missing_value_strategy": Strategy for handling missing values
              (e.g., "drop" to remove rows with NaN)
            - "numerical_column_names": List of columns to convert to numeric type
            - "target_column_name": Name of the target variable column (e.g., "num")

    Returns:
        pd.DataFrame: Cleaned dataset with the following transformations applied:
            - Missing values handled according to strategy
            - All specified columns converted to numeric types
            - New binary "target" column (0=no disease, 1=disease present)

    Side Effects:
        Logs cleaning progress including:
        - Initial and final data shapes
        - Missing value counts before and after cleaning
        - Warnings for values coerced to NaN during type conversion

    Examples:
        >>> cfg = {
        ...     "missing_value_strategy": "drop",
        ...     "numerical_column_names": ["age", "chol", "trestbps"],
        ...     "target_column_name": "num"
        ... }
        >>> raw_data = load_dataset()
        >>> cleaned = clean_data(raw_data, cfg)
        >>> "target" in cleaned.columns
        True

    Note:
        The original target column is preserved. The new "target" column binarizes
        the original multi-class severity (0-4) into binary classification (0/1).
        Type conversion uses "coerce" mode, converting invalid values to NaN.
    """
    # configure logger
    logger = get_logger(__name__)
    logger.info(f"Initial data shape: {df.shape}")
    logger.info(f"Missing values before cleaning: {df.isnull().sum().sum()}")

    # handle missing values
    if cfg["missing_value_strategy"] == "drop":
        df = df.dropna().copy()

    # type conversions (str -> num)
    for name in cfg["numerical_column_names"]:
        before_nulls = df[name].isnull().sum()
        df[name] = pd.to_numeric(df[name], errors="coerce")
        after_nulls = df[name].isnull().sum()
        if after_nulls > before_nulls:
            logger.warning(
                f"Column {name}: {after_nulls - before_nulls} values coerced to NaN"
            )

    # log after dropping missing values
    logger.info(f"Data shape after cleaning: {df.shape}")
    logger.info(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    # encode target variable (num; heart disease) to 0/1
    target_column = cfg["target_column_name"]
    df["target"] = (df[target_column] > 0).astype(int)

    return df


def validate_data(df, cfg):
    """Validate cleaned dataset meets quality and schema requirements.

    Performs comprehensive data quality checks to ensure the cleaned dataset is
    ready for machine learning. Raises exceptions if any validation checks fail,
    preventing downstream pipeline steps from proceeding with invalid data.

    Args:
        df (pd.DataFrame): Cleaned dataset to validate. Should have been processed
            by clean_data() and contain a binary "target" column.
        cfg (dict): Configuration dictionary containing validation criteria:
            - "numerical_column_names": Columns that must have numeric dtype
            - "categorical_column_names": Categorical columns (must be numeric codes)
            - "reasonable_ranges": Dict mapping column names to {"min": x, "max": y}
            - "minimum_rows": Minimum required number of rows in dataset

    Raises:
        ValueError: If any validation check fails:
            - Missing values found in dataset
            - Numerical columns have non-numeric dtypes
            - Categorical columns have non-numeric dtypes
            - "target" column missing or not binary (0/1)
            - Column values outside reasonable ranges
            - Dataset has fewer rows than minimum threshold

    Side Effects:
        Logs success messages for each validation check that passes.

    Examples:
        >>> cfg = {
        ...     "numerical_column_names": ["age", "chol"],
        ...     "categorical_column_names": ["sex", "cp"],
        ...     "reasonable_ranges": {"age": {"min": 0, "max": 120}},
        ...     "minimum_rows": 100
        ... }
        >>> cleaned_df = clean_data(raw_df, cfg)
        >>> validate_data(cleaned_df, cfg)  # Passes silently if valid
        >>> # Or raises ValueError with descriptive message if invalid

    Note:
        This function performs no transformations - it only validates. If validation
        fails, the original cleaning logic should be reviewed. Categorical columns
        are expected to already be encoded as numeric codes (not strings).
    """
    # configure logger
    logger = get_logger(__name__)

    # check no missing values remain
    if df.isnull().sum().sum() != 0:
        raise ValueError(f"Too many missing values: {df.isnull().sum().sum()} missing")
    logger.info("Missing values check passed")

    # check numerical columns are numerical
    for col in cfg["numerical_column_names"]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"Numerical column {col} contains '{type(df[col])}' entries"
                )

    # check categorical codes are integers (not NaN)
    for col in cfg["categorical_column_names"]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"Categorical column {col} contains '{type(df[col])}' entries; should be numerical"
                )
    logger.info("Numerical check passed")

    # check target column exists and is binary
    try:
        df["target"]
    except KeyError:
        raise ValueError("The column 'target' does not exist")

    if not df["target"].isin([0, 1]).all():
        raise ValueError("The column 'target' is not binary")

    logger.info("Target column check passed")

    # check for reasonable ranges
    for col, bounds in cfg["reasonable_ranges"].items():
        if col in df.columns:
            if (df[col] < bounds["min"]).any() or (df[col] > bounds["max"]).any():
                raise ValueError(
                    f"Column {col} has values outside reasonable range"
                    f"[{bounds['min']}, {bounds['max']}]"
                )
    logger.info("Reasonable ranges check passed")

    # check enough data still exists for training
    if len(df) < cfg["minimum_rows"]:
        raise ValueError(
            f"Less than {cfg['minimum_rows']} rows left in dataset"
            f"[{df.sum(axis=1)}]"
        )
    logger.info("Row count check passed")


def main():
    """Command-line interface for data cleaning and validation pipeline.

    Orchestrates the complete data cleaning workflow by loading the raw dataset,
    applying cleaning transformations, validating the results, and saving the
    cleaned data to CSV. This is a standalone script for data preprocessing.

    Side Effects:
        - Loads raw dataset from default path
        - Loads configuration from config file
        - Writes cleaned data to CSV at configured processed data path
        - Logs all pipeline steps and their success/failure status

    Raises:
        ValueError: If data validation fails (propagated from validate_data)
        FileNotFoundError: If raw data or config file cannot be found
        IOError: If writing to CSV fails

    Examples:
        $ python clean.py
        # Executes full cleaning pipeline with default paths

    Note:
        This function uses default paths configured in load_dataset() and
        load_config(). For programmatic use with custom paths, call
        clean_data() and validate_data() directly instead.
    """
    # configure logger
    logger = get_logger(__name__)

    # get df
    df = load_dataset()

    # get cfg
    cfg = load_config()

    # log successful finding
    logger.info("Configuration parameters found")

    # clean data
    clean = clean_data(df, cfg)
    logger.info(f"Dataset cleaned successfully")

    # validate cleaned data
    validate_data(clean, cfg)
    logger.info("Clean dataset validated")

    # configure file path
    processed_file_path = (
        Path(cfg["paths"]["processed_data"]["folder"])
        / cfg["paths"]["processed_data"]["file"]
    )

    # write to csv
    clean.to_csv(processed_file_path)
    logger.info(f"Clean data written to {processed_file_path}")


if __name__ == "__main__":
    main()
