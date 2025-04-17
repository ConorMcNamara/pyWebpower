from scipy.optimize import bisect

import numpy as np


def nuniroot(f, low_val: float = 0, high_val: float = 1, max_length: int = 100) -> float:
    """Calculates the root of our function f given low_val and high_val.

    Parameters
    ----------
    f: function
        The function we are applying our bisect method on
    low_val: float, default=0
        The low end of our interval for bisection
    high_val: float, default=1
        The high end of our interval for bisection
    max_length: int, default=100
        How many intervals between low_val and high_val we will have

    Returns
    -------
    The root of our function given low_val and high_val
    """
    x = np.linspace(low_val, high_val, max_length)
    f_output = np.array([f(x_i) for x_i in x])
    if min(f_output) * max(f_output) > 0:
        raise ValueError(
            "The specified parameters do not yield valid results. Please try to supply a different interval, e.g., "
            "using interval=[0, 1], for your parameter."
        )
    else:
        low = max(f_output[f_output < 0])
        high = min(f_output[f_output > 0])
        interval = [x[f_output == low][0], x[f_output == high][0]]
        return bisect(f, interval[0], interval[1])
