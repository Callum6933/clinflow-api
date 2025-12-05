from clinflow.logging_utils import get_logger
from clinflow.config import load_config
from pathlib import Path
import pandas as pd


def load_dataset(filepath=None):
    # configure logger
    logger = get_logger(__name__)

    # get config
    cfg = load_config()

    # define path to dataset
    if filepath is None:
        path = (
            Path(cfg["paths"]["raw_data"]["folder"]) / cfg["paths"]["raw_data"]["file"]
        )
    else:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Filepath does not exist: {filepath}")

    # load dataset into pandas dataframe
    df = pd.read_csv(path)
    logger.info("Dataset successfully loaded")
    return df


def main():
    df = load_dataset()

    # print head and shape
    print(f"{df.head()}\n")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
