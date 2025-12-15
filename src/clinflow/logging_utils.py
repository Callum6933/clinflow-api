import logging
from pathlib import Path
from clinflow.config import load_config


def get_logger(name):
    """Create or retrieve a configured logger with console and file output.

    Sets up a logger with dual output: console (stdout) and file. The logger is
    configured with DEBUG level and a standardized format. This function is
    idempotent - calling it multiple times with the same name returns the same
    logger instance without adding duplicate handlers.

    Args:
        name (str): Name for the logger, typically __name__ of the calling module.
            This appears in log messages to identify the source.

    Returns:
        logging.Logger: Configured logger instance with two handlers:
            - StreamHandler: Outputs to console (stdout) at DEBUG level
            - FileHandler: Writes to log file at DEBUG level

    Side Effects:
        - Creates logs directory if it doesn't exist (from config)
        - Creates/appends to log file specified in configuration
        - Registers logger in Python's global logging registry

    Examples:
        >>> # Typical usage in a module
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        2025-12-16 10:30:45 - my_module - INFO - Processing started
        >>>
        >>> # Using in different modules
        >>> logger = get_logger("data.pipeline")
        >>> logger.debug("Debug information")
        >>> logger.error("An error occurred")

    Note:
        - Log format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        - Both handlers use DEBUG level, so all messages are logged
        - Log file path is read from configuration: cfg["paths"]["logging"]["file"]
        - The function checks for existing handlers to prevent duplicates
        - All loggers created by this function share the same configuration
    """
    cfg = load_config()

    # create logs directory if it doesn't exist
    logs_dir = Path(cfg["paths"]["logging"]["folder"])
    logs_file = Path(cfg["paths"]["logging"]["file"])
    logs_filepath = logs_dir / logs_file

    logs_dir.mkdir(exist_ok=True, parents=True)

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create file handler
        fh = logging.FileHandler(logs_filepath)
        fh.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # add formatter both handlers
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add both handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
