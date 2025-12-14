from clinflow.data.load import load_dataset
from clinflow.data.query import query_patients
from clinflow.models.train import train_model
from clinflow.models.train import evaluate_model
from clinflow.models.io import save_model
from clinflow.logging_utils import get_logger
from clinflow.config import load_config
from pathlib import Path


def main():
    import argparse

    logger = get_logger(__name__)
    cfg = load_config()

    # initialise argument parser
    parser = argparse.ArgumentParser(description="Train the clinflow risk model")

    # add mutually exclusive arguments for input filepath
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--csv", metavar="PATH", help="Path to processed CSV file"
    )
    source_group.add_argument(
        "--from-db", action="store_true", help="Load data from SQLite database"
    )

    # add arguments for optional arguments (output path, db query, db boolean)
    parser.add_argument("--output-path", "-op", type=str, default=None)
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="all",
        help="Only use this flag with '--from-db'",
    )
    # parse arguments
    args = parser.parse_args()
    logger.info("Command line arguments parsed")

    # run full pipeline (load -> train -> eval -> save)
    # 1. load data from csv or database
    if args.from_db:
        df = query_patients(args.query)
        logger.info("Dataset loaded from db")
    elif args.csv:
        path_to_clean_data = Path(args.csv)
        try:
            df = load_dataset(path_to_clean_data)
        except Exception as e:
            logger.error(f"Error: input path {args.csv} not valid ({e})")
            raise
        logger.info("Dataset loaded from csv")

    # 2. train model
    model = train_model(df, cfg)
    logger.info("Model trained successfully")

    # 3. evaluate model
    y_test = model["y_test"]
    y_pred = model["y_pred"]

    evals_path = evaluate_model(y_test, y_pred)
    logger.info(f"Model evals calculated and saved to '{evals_path}'")

    # 4. save model
    path = save_model(model, args.output_path)
    logger.info(f"Model saved to '{path}'")
    logger.info("Pipeline completed successfully")
    return


if __name__ == "__main__":
    main()
