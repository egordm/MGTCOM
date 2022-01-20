import numpy as np


def drop_infna(df):
    """
    Drop rows with inf or nan values
    """
    return df.replace([np.PINF, np.NINF], np.nan).dropna()