import pandas as pd
import sqlite3
from clinflow.logging_utils import get_logger

logger = get_logger(__name__)

DATABASE_PATH = "data/clinflow.db"

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
    """run a preset query and return results."""
    logger.info(f"Querying preset: '{preset}'")

    filter_spec = QUERY_SPECS[preset]
    where_clause, params = build_where_clause(filter_spec)
    query = f"SELECT * FROM patients WHERE {where_clause}"
    logger.debug(f"SQL query: {query} with params: {params}")

    con = sqlite3.connect(DATABASE_PATH)
    results = pd.read_sql_query(query, con, params=params)

    logger.info(f"Query returned {len(results)} rows")
    con.close()
    return results


def build_where_clause(filter_spec):
    """build a SQL WHERE clause"""
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
    import argparse

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
