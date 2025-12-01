from clinflow.logging_utils import get_logger
from ucimlrepo import fetch_ucirepo
from pathlib import Path
import os


def download_dataset():
    # configure logger
    logger = get_logger(__name__)

    # define download path
    current_file = Path(__file__)
    root = current_file.parent.parent.parent.parent
    path = root / "data" / "raw" / "heart_disease_dataset.csv"

    # check if dataset is already downloaded
    if os.path.isfile(path):
        logger.info(f"Dataset already exists at {path}")
        return

    # fetch dataset
    heart_disease = fetch_ucirepo(name="Heart Disease")

    # data (as pandas dataframe)
    df = heart_disease.data.original
    df.to_csv(path_or_buf=path)
    logger.info(f"Dataset downloaded as {path}")


def main():
    download_dataset()


if __name__ == "__main__":
    main()
