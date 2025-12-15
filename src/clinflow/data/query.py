import pandas as pd
import sqlite3
from pathlib import Path
from clinflow.logging_utils import get_logger
from clinflow.config import load_config

QUERY_SPECS = {
    "high_risk_seniors": {
        "age": {"min": 60},
        "target": {"equals": 1},
    },
    "young_with_high_chol": {
        "age": {"max": 40},
        "chol": {"min": 200},
    },
    "exercise_induced_angina": {
        "exang": {"equals": 1},
    },
}


def query_patients(preset):
    """Query patient data from SQLite database using predefined filter presets.

    This function retrieves patient records from the clinflow database by applying
    filters defined in the QUERY_SPECS dictionary. If the preset doesn't exist or
    is set to "all", it returns all patient records.

    Args:
        preset (str): Name of the query preset to apply. Must be a key in QUERY_SPECS
            (e.g., "high_risk_seniors", "young_with_high_chol") or "all" to retrieve
            all records. If an invalid preset is provided, defaults to returning all.

    Returns:
        pd.DataFrame: DataFrame containing patient records that match the filter
            criteria. Columns correspond to the patients table schema.

    Examples:
        >>> # Query high-risk senior patients
        >>> seniors = query_patients("high_risk_seniors")
        >>>
        >>> # Query all patients
        >>> all_patients = query_patients("all")
        >>>
        >>> # Invalid preset falls back to all patients
        >>> results = query_patients("nonexistent")  # Logs warning, returns all

    Note:
        The database connection is automatically opened and closed within this function.
        Queries use parameterized statements to prevent SQL injection.
    """
    logger = get_logger(__name__)

    logger.info(f"Querying preset: '{preset}'")
    cfg = load_config()

    # connect to SQL databse
    database_path = (
        Path(cfg['paths']['database_path']['folder'])
        / cfg['paths']['database_path']['file']
    )
    with sqlite3.connect(database_path) as con:
        # build SQL query from presets
        if preset in QUERY_SPECS:
            filter_spec = QUERY_SPECS[preset]
            where_clause, params = build_where_clause(filter_spec)
            query = f"SELECT * FROM patients WHERE {where_clause}"
            logger.info(f"SQL query: {query} with params: {params}")

            results = pd.read_sql_query(query, con, params=params)
        else:
            if not preset == "all":
                logger.warning(f"Preset '{preset}' not found: returning all")
            query = "SELECT * FROM patients"
            results = pd.read_sql_query(query, con)
            logger.info(f"SQL query: '{query}'")

        logger.info(f"Query returned {len(results)} rows")

    return results


def build_where_clause(filter_spec):
    """Build a parameterized SQL WHERE clause from a filter specification.

    Constructs a SQL WHERE clause with parameterized placeholders (?) to safely
    filter database queries. Supports min, max, and equals operations on columns.

    Args:
        filter_spec (dict): Dictionary mapping column names to filter rules.
            Each rule is a dict that can contain:
            - "min": Include records where column >= this value
            - "max": Include records where column <= this value
            - "equals": Include records where column == this value
            Multiple rules for the same column are combined with AND.

    Returns:
        tuple[str, list]: A tuple containing:
            - str: SQL WHERE clause with ? placeholders (e.g., "age >= ? AND chol <= ?")
            - list: Ordered list of parameter values to substitute for placeholders

    Examples:
        >>> spec = {"age": {"min": 60}, "chol": {"max": 200}}
        >>> where_clause, params = build_where_clause(spec)
        >>> print(where_clause)
        age >= ? AND chol <= ?
        >>> print(params)
        [60, 200]

    Note:
        Returns empty strings and lists if filter_spec is empty. This function
        uses parameterized queries to prevent SQL injection attacks.
    """
    conditions = []
    params = []
    for column, rules in filter_spec.items():
        if "min" in rules:
            conditions.append(f"{column} >= ?")
            params.append(rules["min"])
        if "max" in rules:
            conditions.append(f"{column} <= ?")
            params.append(rules["max"])
        if "equals" in rules:
            conditions.append(f"{column} = ?")
            params.append(rules["equals"])
    return " AND ".join(conditions), params


def main():
    """Command-line interface for querying patient data.

    Provides CLI access to patient query functionality with support for:
    - Running predefined query presets (--preset)
    - Listing available query presets (--list-presets)

    Command-line Arguments:
        --preset: Name of query preset to run (default: "high_risk_seniors")
        --list-presets: Display all available query presets and their filter rules

    Examples:
        $ python query.py --preset high_risk_seniors
        $ python query.py --preset all
        $ python query.py --list-presets
    """
    import argparse

    logger = get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="high_risk_seniors")
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args()

    # lists available CLI presets
    if args.list_presets:
        logger.info(f"Listing {len(QUERY_SPECS)} available presets")
        for spec_name, spec_details in QUERY_SPECS.items():
            print(f"Spec: {spec_name}")
            print(f"Details: {spec_details}")
            print("-" * 30)
        return
    results = query_patients(args.preset)
    print(results)


if __name__ == "__main__":
    main()
