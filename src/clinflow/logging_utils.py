import logging
import os


def get_logger(name):
    # create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create file handler
        fh = logging.FileHandler("logs/clinflow.log")
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
