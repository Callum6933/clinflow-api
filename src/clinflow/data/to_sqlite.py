from clinflow.data.load import load_dataset
from clinflow.logging_utils import get_logger
from clinflow.config import load_config
from pathlib import Path
import sqlite3


def write_to_SQL_db(df=None):
    logger = get_logger(__name__)
    cfg = load_config()

    # load data if not provided
    if df is None:
        path_to_clean_data = (
            Path(cfg["paths"]["processed_data"]["folder"])
            / cfg["paths"]["processed_data"]["file"]
        )
        df = load_dataset(path_to_clean_data)
        logger.info(f"Loaded data from {path_to_clean_data}")
    else:
        logger.info("Using provided DataFrame")
    path_to_db = (
        Path(cfg["paths"]["database_path"]["folder"])
        / cfg["paths"]["database_path"]["file"]
    )

    try:
        # insert clean.csv rows into clinflow.df
        con = sqlite3.connect(path_to_db)
        df.to_sql("patients", con, if_exists="replace", index=False)

        # verification to confirm
        # * database file created successfully
        if not path_to_db.exists():
            logger.error(f"Database does not exist at {path_to_db}")
            raise ValueError

        logger.info(f"db exists at {path_to_db}")

        # * row count matches cleaned csv
        cur = con.cursor()

        db_row_count = cur.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        if len(df) != db_row_count:
            raise ValueError(
                f"db row count ({db_row_count}) does NOT match csv row count ({len(df)})"
            )

        logger.info(f"db row count ({db_row_count}) matches df row count ({len(df)})")

    finally:
        if "con" in locals():
            con.close()


def main():
    write_to_SQL_db()


if __name__ == "__main__":
    main()
