from multiprocessing import Process, Manager
import time
import logging

logging.basicConfig(level=logging.DEBUG)


# This function wrapped could be used to generate multiple parallel workers
# but we use it to ensure we can get the return metric value
def _func_wrapper(func, procnum, return_dict):
    return_dict[procnum] = func()


class AsyncModelTimeout():
    """This class can wrap a model training function and ensure it stops in a specified timeout."""

    def __init__(self, target_function, timeout_in_seconds, logging_name=None):
        self.target_function = target_function,
        self.timeout_in_seconds = timeout_in_seconds
        self.logger = logging.getLogger('AsyncModelTimeout' if logging_name is None else logging_name)

    def run(self):
        manager = Manager()
        return_dict = manager.dict()
        action_process = Process(target=_func_wrapper, args=(self.target_function[0], 0, return_dict))
        action_process.start()
        action_process.join(timeout=self.timeout_in_seconds)
        if action_process.is_alive():
            action_process.terminate()
            self.logger.warning("Model timed out. Terminated.")
            return False, None  # No metric exists, so we return None
        else:
            self.logger.info("Model completed successfully.")
            return True, return_dict[0]
