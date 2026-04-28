import logging
import sys

_FMT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_FMT,
    datefmt=_DATE_FMT,
    stream=sys.stdout,
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
