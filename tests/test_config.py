from clinflow.config import load_config
from pathlib import Path

def test_returns_dictionary():
    assert isinstance(load_config(), dict)

def test_keys():
    cfg = load_config()
    raw_path = Path(cfg["paths"]["raw_data"]["folder"]) / cfg["paths"]["raw_data"]["file"]
    processed_path = Path(cfg["paths"]["processed_data"]["folder"]) / cfg["paths"]["processed_data"]["file"]
    target_column = cfg["target_column_name"]
    
    assert raw_path.exists()
    assert processed_path.exists()
    assert target_column == "num"

def test_custom_path():
    assert isinstance(load_config("config/config.yml"), dict)


