from typing import Callable
from scipy.stats import loguniform


def float_to_int(rvs: Callable) -> Callable:
    """
    Wraps a callable that generates floating-point random values and converts
    its output to rounded integers.

    Args:
        rvs (Callable): A callable that generates random floating-point values.

    Returns:
        Callable: A wrapped callable that returns rounded integers.
    """

    def rvs_wrapper(*args, **kwargs):
        """
        Wrapper function to generate rounded integer values.

        Args:
            *args: Positional arguments to pass to the original callable.
            **kwargs: Keyword arguments to pass to the original callable.

        Returns:
            int: Rounded integer values generated by the original callable.
        """
        return rvs(*args, **kwargs).round().astype(int)

    return rvs_wrapper


def int_loguniform(low: float, high: float) -> loguniform:
    """
    Creates a loguniform distribution where the random variates (rvs) are rounded to integers.

    Args:
        low (float): The lower bound of the loguniform distribution.
        high (float): The upper bound of the loguniform distribution.

    Returns:
        loguniform: A loguniform distribution with integer random variates.
    """
    # Create a loguniform object
    lu = loguniform(low, high)
    # Wrap its rvs method with float_to_int
    lu.rvs = float_to_int(lu.rvs)
    # Return modified loguniform object
    return lu
