from clinflow.logging_utils import get_logger
from pathlib import Path


def run_data_pipeline():
    from clinflow.data.load import load_dataset
    from clinflow.data.clean import clean_data, validate_data
    from clinflow.config import load_config
    from clinflow.data.to_sqlite import write_to_SQL_db

    logger = get_logger(__name__)

    # load raw data
    try:
        raw = load_dataset()
    except Exception as e:
        raise
    logger.info("Raw data loaded successfully")

    # load config file
    try:
        cfg = load_config()
    except Exception as e:
        raise
    logger.info("Config file loaded successfully")

    # clean data
    try:
        clean = clean_data(raw, cfg)
    except Exception as e:
        raise
    logger.info("Data cleaned successfully")

    # validate data
    try:
        validate_data(clean, cfg)
    except Exception as e:
        raise
    logger.info("Clean data validated")

    # write to csv
    try:
        processed_file_path = (
            Path(cfg["paths"]["processed_data"]["folder"])
            / cfg["paths"]["processed_data"]["file"]
        )
        clean.to_csv(processed_file_path)
    except Exception as e:
        raise
    logger.info("File successfully written to csv at {processed_file_path}/")

    # store to SQL database
    try:
        write_to_SQL_db(clean)
    except Exception as e:
        raise
    logger.info("Clean data successfully written to sqlite db")


def main():
    logger = get_logger(__name__)
    run_data_pipeline()
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
