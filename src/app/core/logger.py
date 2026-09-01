import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        # Always log to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

        # Optionally also log to a file
        if os.getenv("LOG_TO_FILE", "") == "true":
            file_handler = logging.FileHandler("app.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
