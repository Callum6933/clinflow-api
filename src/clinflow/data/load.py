from clinflow.logging_utils import get_logger
from pathlib import Path
import pandas as pd


def load_raw_data(filepath=None):
    # configure logger
    logger = get_logger(__name__)
    
    # define path to dataset
    if filepath is None:
        current_file = Path(__file__)
        root = current_file.parent.parent.parent.parent
        path = root / "data" / "raw" / "heart_disease_dataset.csv"
    else:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Filepath does not exist: {filepath}")

    # load dataset into pandas dataframe
    df = pd.read_csv(path)
    logger.info("Dataset successfully loaded")
    return df


def main():
    df = load_raw_data()

    # print head and shape
    print(f"{df.head()}\n")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
