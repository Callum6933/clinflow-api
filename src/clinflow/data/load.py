from pathlib import Path
import pandas as pd

def load_raw_data():
    # define path to dataset
    current_file = Path(__file__)
    root = current_file.parent.parent.parent.parent
    path = root / "data" / "raw" / "heart_disease_dataset.csv"

    # load dataset into pandas dataframe
    df = pd.read_csv(path)
    
    # print head
    print(f"{df.head()}\n")

    # print shape
    print(f"Shape: {df.shape}")

    return(df)

def main():
    load_raw_data()

if __name__ == "__main__":
    main()