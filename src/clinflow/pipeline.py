from clinflow.logging_utils import get_logger


def run_data_pipeline():
    from clinflow.data.load import load_raw_data
    from clinflow.data.clean import clean_data, validate_data
    from clinflow.config import load_config
    from clinflow.data.to_sqlite import write_to_SQL_db

    logger = get_logger(__name__)
    
    # load raw data
    try:
        raw = load_raw_data()
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
        processed_file_path = cfg["path_to_processed_data"]
        clean.to_csv(f"{processed_file_path}/clean.csv")
    except Exception as e:
        raise
    logger.info("File successfully written to csv at {processed_file_path}/")

    # store to SQL database
    try:
        write_to_SQL_db()
    except Exception as e:
        raise
    logger.info("Clean data successfully written to sqlite db")

def main():
    logger = get_logger(__name__)
    run_data_pipeline()
    logger.info("Pipeline completed successfully")
    


if __name__ == "__main__":
    main()