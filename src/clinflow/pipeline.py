from clinflow.logging_utils import get_logger
from pathlib import Path


def run_data_pipeline():
    """Execute the complete data processing pipeline from raw data to storage.

    This function orchestrates the entire data pipeline workflow, performing the
    following steps in sequence:
    1. Load raw dataset from configured source
    2. Load configuration settings
    3. Clean and transform data according to config rules
    4. Validate cleaned data meets quality standards
    5. Write processed data to CSV file
    6. Store processed data in SQLite database

    The pipeline uses configuration settings from the config file to determine
    data paths, cleaning rules, and validation criteria. All operations are
    logged for monitoring and debugging purposes.

    Raises:
        FileNotFoundError: If raw data file or config file cannot be found.
        ValueError: If data validation fails after cleaning.
        IOError: If CSV write or database write operations fail.

    Side Effects:
        - Writes processed data to CSV file at the configured path
        - Updates/creates SQLite database with cleaned data
        - Generates log messages at INFO level for each pipeline step

    Examples:
        >>> run_data_pipeline()
        # Executes full pipeline with default configuration
    """
    from clinflow.data.load import load_dataset
    from clinflow.data.clean import clean_data, validate_data
    from clinflow.config import load_config
    from clinflow.data.to_sqlite import write_to_SQL_db

    logger = get_logger(__name__)

    # load raw data
    raw = load_dataset()
    logger.info("Raw data loaded successfully")

    # load config file
    cfg = load_config()
    logger.info("Config file loaded successfully")

    # clean data
    clean = clean_data(raw, cfg)
    logger.info("Data cleaned successfully")

    # validate data
    validate_data(clean, cfg)
    logger.info("Clean data validated")

    # write to csv
    processed_file_path = (
        Path(cfg["paths"]["processed_data"]["folder"])
        / cfg["paths"]["processed_data"]["file"]
    )
    clean.to_csv(processed_file_path)
    logger.info(f"File successfully written to csv at {processed_file_path}")

    # store to SQL database
    write_to_SQL_db(clean)
    logger.info("Clean data successfully written to sqlite db")


def main():
    logger = get_logger(__name__)
    run_data_pipeline()
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
