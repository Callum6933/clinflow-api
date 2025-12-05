from clinflow.logging_utils import get_logger
from ucimlrepo import fetch_ucirepo
from clinflow.config import load_config
from clinflow.data.clean import validate_data
from pathlib import Path


def download_dataset(
    dataset_name: str = "Heart Disease", output_path: Path = None
) -> Path:
    """
    Download UCI Heart Disease Dataset to local storage

    Fetches the Heart Disease dataset from UCI Machine Learning Repository
    and saves it as a CSV file. Skips download if file already exists

    Returns:
        Path: Path to the downlaoded CSV file

    Raises:
        ConnectionError: If unable to reach UCI repository
        IOError: If unable to write to disk

    Example:
        >>> path = download_dataset()
        >>> print(f"Data saved to: {path})
    """

    # configure logger
    logger = get_logger(__name__)

    # load config file
    cfg = load_config()

    # define download path
    download_path = (
        Path(cfg["paths"]["raw_data"]["folder"]) / cfg["paths"]["raw_data"]["file"]
    )

    # check if dataset is already downloaded
    if download_path.exists():
        logger.info(f"Dataset already exists at {download_path}")
        return download_path

    # fetch dataset
    try:
        heart_disease = fetch_ucirepo(name=dataset_name)
    except Exception as e:
        logger.error(f"Failed to fetch heart disease data: {e}")
        raise

    # data (as pandas dataframe)
    df = heart_disease.data.original

    # validate data
    try:
        validate_data(df, cfg)
    except Exception as e:
        logger.error(f"Downloaded dataset is not valid: {print(df)}")
        raise

    try:
        download_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(download_path, index=False)
    except Exception as e:
        logger.error("Couldn't convert heart disease dataframe to csv")
        raise

    logger.info(f"Dataset downloaded as {download_path}")
    return download_path


def main():
    download_dataset()


if __name__ == "__main__":
    main()
