import random

import numpy as np


def set_random_seed(seed):
    """
    Set the random seed for all random functions.
    """
    random.seed(seed)
    np.random.seed(seed)
