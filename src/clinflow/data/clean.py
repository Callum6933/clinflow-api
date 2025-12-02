from clinflow.data.load import load_raw_data
from clinflow.logging_utils import get_logger
import yaml
import pandas as pd


def clean_data(df, cfg):
    # configure logger
    logger = get_logger(__name__)
    logger.info(f"Initial data shape: {df.shape}")
    logger.info(f"Missing values before cleaning: {df.isnull().sum().sum()}")

    # handle missing values
    if cfg["missing_value_strategy"] == "drop":
        df = df.dropna()

    # type conversions (str -> num)
    for key in cfg["numerical_column_names"]:
        before_nulls = df[key].isnull().sum()
        df[key] = pd.to_numeric(df[key], errors="coerce")
        after_nulls = df[key].isnull().sum()
        if after_nulls > before_nulls:
            logger.warning(
                f"Column {key}: {after_nulls - before_nulls} values coerced to NaN"
            )

    # convert categorical to numerical codes
    for key in cfg["categorical_column_names"]:
        df[key] = df[key].astype("category").cat.codes

    # log after dropping missing values
    logger.info(f"Data shape after cleaning: {df.shape}")
    logger.info(f"Missing values after cleaning: {df.isnull().sum().sum()}")


    # encode target variable (num; heart disease) to 0/1
    target_column = cfg["target_column_name"]
    df["target"] = (df[target_column] > 0).astype(int)

    return df


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

    # configure file path
    processed_file_path = configuration["path_to_processed_data"]

    # write to csv
    clean.to_csv(f"{processed_file_path}/clean.csv")
    logger.info(f"Clean data written to {processed_file_path}")


if __name__ == "__main__":
    main()
