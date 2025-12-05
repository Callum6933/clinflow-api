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


def print_metrics(path=None, df=None):
    # configure logger
    logger = get_logger(__name__)

    # records/features count
    df = load_dataset(path)
    logger.info(f"Records: {df.shape[0]}")
    logger.info(f"Features: {df.shape[1]}")

    # columns with (and no. of) missing data
    logger.info(f"Missing values:\n{df.isnull().sum()}")

    # target distribution
    # - number of individuals with and without heart disease
    target_distribution = df["num"].value_counts()
    logger.info(f"Without heart disease: {target_distribution[0]}")
    logger.info(f"With heart disease: {target_distribution[1:].sum()}")

    # - percentage of individuals with each severity of heart disease
    logger.info(
        f"Percentages:\nNo heart disease: {round((target_distribution[0]/target_distribution.sum()), 2) * 100}%"
    )
    logger.info(
        f"Severity 1: {round((target_distribution[1]/target_distribution.sum()), 2) * 100}%"
    )
    logger.info(
        f"Severity 2: {round((target_distribution[2]/target_distribution.sum()), 2) * 100}%"
    )
    logger.info(
        f"Severity 3: {round((target_distribution[3]/target_distribution.sum()), 2) * 100}%"
    )
    logger.info(
        f"Severity 4: {round((target_distribution[4]/target_distribution.sum()), 2) * 100}%"
    )


def main():
    # parse command line arguments
    file_path = parse_cli_args()

    # load dataset and print metrics
    print_metrics(file_path)


if __name__ == "__main__":
    main()
