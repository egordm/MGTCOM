import logging.config
import os
import threading

import yaml

from shared.constants import GLOBAL_CONFIG_PATH

LOGGING_CONFIG = GLOBAL_CONFIG_PATH.joinpath('logging.yml')

# with open(LOGGING_CONFIG, 'r') as f:
#     config = yaml.safe_load(f.read())
#     logging.config.dictConfig(config)
#     logging.captureWarnings(True)


def get_logger(name: str):
    """Logs a message
    Args:
    name(str): name of logger
    """
    logger = logging.getLogger(name)
    return logger


def get_logpipe(name: str, level=logging.INFO):
    return LogPipe(get_logger(name), level)


class LogPipe(threading.Thread):
    def __init__(self, logger: logging.Logger, level=logging.INFO):
        """Setup the object with a logger and a loglevel
        and start the thread
        """
        threading.Thread.__init__(self)
        self.daemon = False
        self.level = level
        self.logger = logger
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)
        self.start()

    def fileno(self):
        """Return the write file descriptor of the pipe
        """
        return self.fdWrite

    def run(self):
        """Run the thread, logging everything.
        """
        for line in iter(self.pipeReader.readline, ''):
            self.logger.log(self.level, line.strip('\n'))

        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe.
        """
        os.close(self.fdWrite)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
