import logging

logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s', level=logging.INFO)


def get_logger(name: str):
    """Logs a message
    Args:
    name(str): name of logger
    """
    logger = logging.getLogger(name)
    return logger
