from clinflow.data.load import load_raw_data
from clinflow.logging_utils import get_logger
from pathlib import Path
import sqlite3
import yaml


def write_to_SQL_db():
    # configure logger
    logger = get_logger(__name__)

    # configure path to clean.csv
    with open("config/config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    path_to_clean_data = cfg["path_to_processed_data"]

    # load data
    df = load_raw_data(f"{path_to_clean_data}/clean.csv")

    # insert clean.csv rows into clinflow.df
    con = sqlite3.connect(f"data/clinflow.db")
    df.to_sql("patients", con, if_exists="replace", index=False)

    # verification to confirm
    # * database file created successfully
    path = Path(f"data/clinflow.db")
    if not path.exists():
        raise ValueError("db does not exist at {path}")

    logger.info(f"db exists at {path}")

    # * row count matches cleaned csv
    cur = con.cursor()

    db_row_count = cur.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    if len(df) != db_row_count:
        raise ValueError(
            f"db row count ({db_row_count}) does NOT match csv row count ({len(df)})"
        )

    logger.info(f"db row count ({db_row_count}) matches df row count ({len(df)})")

    con.close()


def main():
    write_to_SQL_db()


if __name__ == "__main__":
    main()
