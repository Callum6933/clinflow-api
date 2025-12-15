from clinflow.data.load import load_dataset
from clinflow.logging_utils import get_logger
import argparse
import pandas as pd


def parse_cli_args():
    # parse cli input
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the dataset CSV file", default=None)
    args = parser.parse_args()
    return args.path


def print_metrics(path=None):
    """Print exploratory data analysis metrics for the heart disease dataset.

    Performs initial data exploration by calculating and logging key dataset
    characteristics including shape, missing values, and target distribution.
    This function is designed for quick command-line data inspection.

    Args:
        path (str, optional): Path to the dataset CSV file. If None, uses the
            default dataset path configured in load_dataset(). Defaults to None.

    Side Effects:
        Logs the following metrics to the configured logger:
        - Total number of records (rows)
        - Total number of features (columns)
        - Missing value counts for each column
        - Target variable ("num") distribution as percentages, showing:
            * No heart disease (num=0)
            * Severity levels 1-4 (num=1,2,3,4)

    Examples:
        >>> # Use default dataset path
        >>> print_metrics()
        INFO: Records: 303
        INFO: Features: 14
        INFO: Missing values:
        age       0
        sex       0
        ...
        INFO: Percentages:
        INFO:   No heart disease: 45.54%
        INFO:   Severity 1: 18.15%

        >>> # Use custom dataset path
        >>> print_metrics(path="data/raw/custom_heart_data.csv")

    Note:
        This function does not return any values; all output is logged.
        The target variable "num" represents heart disease severity where
        0 = no disease and 1-4 = increasing severity levels.
    """
    # configure logger
    logger = get_logger(__name__)

    # records/features count
    df = load_dataset(path)
    logger.info(f"Records: {df.shape[0]}")
    logger.info(f"Features: {df.shape[1]}")

    # columns with (and no. of) missing data
    logger.info(f"Missing values:\n{df.isnull().sum()}")

    # target distribution
    target_distribution = df["num"].value_counts().sort_index()
    total = target_distribution.sum()

    # summary counts
    logger.info("Percentages:")
    for severity, count in target_distribution.items():
        pct = round((count / total) * 100, 2)
        label = "No heart disease" if severity == 0 else f"Severity {severity}"
        logger.info(f"  {label}: {pct}%")


def main():
    # parse command line arguments
    file_path = parse_cli_args()

    # load dataset and print metrics
    print_metrics(file_path)


if __name__ == "__main__":
    main()
