import numpy as np


def prng(description: str) -> np.random.Generator:
    """
    Returns an instance of :class:`numpy.random.Generator` seeded based on a text description.

    >>> from pypuf.random import prng
    >>> seed = 5
    >>> my_prng = prng(f'my favorite random numbers, seed {seed}')
    >>> my_prng.integers(2, 6, 3)
    array([4, 3, 2])
    """
    return np.random.default_rng(int.from_bytes(description.encode(), byteorder='big'))


def seed(description: str) -> int:
    """
    Based on the description provided, returns a random integer in the interval of [0, 2**32)
    that can be used as a seed.
    """
    return prng(description).integers(0, 2**32 - 1)
