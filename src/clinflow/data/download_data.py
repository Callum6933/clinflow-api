from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path
import os

def download_dataset():
    # define download path
    current_file = Path(__file__)
    root = current_file.parent.parent.parent.parent
    path = root / "data" / "raw" / "heart_disease_dataset.csv"

    # check if dataset is already downloaded
    if os.path.isfile(path):
        print(f"✅ Dataset already exists at {path}")
        return

    # fetch dataset 
    heart_disease = fetch_ucirepo(name="Heart Disease") 
    
    # data (as pandas dataframe) 
    df = heart_disease.data.original
    df.to_csv(path_or_buf=path)
    

def main():
    download_dataset()

if __name__ == "__main__":
    main()