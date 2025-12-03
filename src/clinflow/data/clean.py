from clinflow.data.load import load_raw_data
from clinflow.logging_utils import get_logger
import pandas as pd
import yaml


def clean_data(df, cfg):
    # configure logger
    logger = get_logger(__name__)
    logger.info(f"Initial data shape: {df.shape}")
    logger.info(f"Missing values before cleaning: {df.isnull().sum().sum()}")

    # handle missing values
    if cfg["missing_value_strategy"] == "drop":
        df = df.dropna()

    # type conversions (str -> num)
    for name in cfg["numerical_column_names"]:
        before_nulls = df[name].isnull().sum()
        df[name] = pd.to_numeric(df[name], errors="coerce")
        after_nulls = df[name].isnull().sum()
        if after_nulls > before_nulls:
            logger.warning(
                f"Column {name}: {after_nulls - before_nulls} values coerced to NaN"
            )

    # convert categorical to numerical codes
    for name in cfg["categorical_column_names"]:
        df[name] = df[name].astype("category").cat.codes

    # log after dropping missing values
    logger.info(f"Data shape after cleaning: {df.shape}")
    logger.info(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    # encode target variable (num; heart disease) to 0/1
    target_column = cfg["target_column_name"]
    df["target"] = (df[target_column] > 0).astype(int)

    return df


def validate_data(df, cfg):
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

    for row in df["target"]:
        if row != 0 and row != 1:
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

    return


def main():
    # configure logger
    logger = get_logger(__name__)

    # get df
    dataset = load_raw_data()

    # get cfg
    try:
        with open("config/config.yml", "r") as f:
            configuration = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Couldn't find file at config/config.yml")
        raise

    # log successful finding
    logger.info("Configuration parameters found")

    # clean data
    clean = clean_data(df=dataset, cfg=configuration)
    logger.info(f"Dataset cleaned successfully")

    # validate cleaned data
    validate_data(df=clean, cfg=configuration)
    logger.info("Clean dataset validated")

    # configure file path
    processed_file_path = configuration["path_to_processed_data"]

    # write to csv
    clean.to_csv(f"{processed_file_path}/clean.csv")
    logger.info(f"Clean data written to {processed_file_path}")


if __name__ == "__main__":
    main()
