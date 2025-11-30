from clinflow.config import load_config
import os

def test_returns_dictionary():
    assert isinstance(load_config(), dict)

def test_keys():
    parsed_dict = load_config()
    raw_path = parsed_dict["path_to_raw_data"]
    processed_path = parsed_dict["path_to_processed_data"]
    target_column = parsed_dict["target_column_name"]
    
    assert os.path.exists(raw_path)
    assert os.path.exists(processed_path)
    assert target_column == "num"

def test_custom_path():
    assert isinstance(load_config("config/config.yml"), dict)


