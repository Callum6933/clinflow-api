import pandas as pd
from pathlib import Path
from clinflow.pipeline import run_data_pipeline
from clinflow.config import load_config

def test_csv_exists():
    # test if clean.csv exists after running the pipeline
    cfg = load_config()
    path = Path(cfg["paths"]["processed_data"]["folder"]) / cfg["paths"]["processed_data"]["file"]

    run_data_pipeline()
    if path.exists() == False:
        raise FileNotFoundError(f"File: {path} not found")
    
def test_csv_valid():
    # check if the cleaned csv has more than 0 rows
    cfg = load_config()
    path = Path(cfg["paths"]["processed_data"]["folder"]) / cfg["paths"]["processed_data"]["file"]

    run_data_pipeline()

    with open(path, "r") as f:
        df = pd.read_csv(f)

    if len(df) < 1:
        raise ValueError(f"File: {path} has too few rows ({len(df)})")
    
def test_db_exists():
    # check if clinflow db exists after running the pipeline
    database_path = Path("data/clinflow.db")
    if database_path.exists() == False:
        raise FileNotFoundError(f"File: {database_path} could not be opened or does not exist")