import logging.config

import yaml

from shared.constants import GLOBAL_CONFIG_PATH

LOGGING_CONFIG = GLOBAL_CONFIG_PATH.joinpath('logging.yml')

with open(LOGGING_CONFIG, 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logging.captureWarnings(True)


def get_logger(name: str):
    """Logs a message
    Args:
    name(str): name of logger
    """
    logger = logging.getLogger(name)
    return logger
