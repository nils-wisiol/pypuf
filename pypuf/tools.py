from numpy import count_nonzero, array, append
from numpy.random import RandomState
import itertools


def random_input(n, random_instance=RandomState()):
    """
    returns a random {-1,1}-vector of length `n`.

    `choice` method of optionally provided PRNG is used.
     If no PRNG provided, a fresh `numpy.random.RandomState`
     instance is used.
    """
    return random_instance.choice((-1, +1), n)


def all_inputs(n):
    """
    returns an iterator for all {-1,1}-vectors of length `n`.
    """
    return itertools.product((-1, +1), repeat=n)


def random_inputs(n, num, random_instance=RandomState()):
    """
    returns an iterator for a random sample of {-1,1}-vectors of length `n` (with replacement).

    If no PRNG provided, a fresh `numpy.random.RandomState` instance is used.
    """
    for i in range(num):
        yield random_input(n, random_instance)


def sample_inputs(n, num):
    """
    returns an iterator for either random samples of {-1,1}-vectors of length `n` if `num` < 2^n,
    and an iterator for all {-1,1}-vectors of length `n` otherwise.
    Note that we return only 2^n vectors even with `num` > 2^n.
    In other words, the output of this function is deterministic if and only if num >= 2^n.
    """
    return random_inputs(n, num) if num < 2**n else all_inputs(n)

def iter_append_last(array_iterator, x):
    """
    returns an iterator for a given iterator of arrays to which element x will be appended
    """
    for array in array_iterator:
        yield append(array, x)

def approx_dist(a, b, num):
    """
    approximate the distance of two functions a, b by evaluating a random set of inputs.
    a, b needs to have eval() method and input_length member.
    :return: probability (randomly uniform x) for a.eval(x) != b.eval(x)
    """
    assert a.n == b.n
    d = 0
    inputs = array(list(random_inputs(a.n, num)))
    return (num - count_nonzero(a.eval(inputs) == b.eval(inputs))) / num

def compare_functions(x, y):
    """
    compares two function on bytecode layer
    :param x: function object
    :param y: function object
    :return: bool
    """
    xc = x.__code__
    yc = y.__code__
    b = xc.co_code == yc.co_code
    # The bytcode maybe differ from each other https://stackoverflow.com/a/20059029
    b &= xc.co_name == yc.co_name
    return b and xc.co_filename == yc.co_filename

class TrainingSet():
    """
    Basic data structure to hold a collection of challenge response pairs.
    Note that this is, strictly speaking, not a set.
    """

    def __init__(self, instance, N):
        self.challenges = array(list(sample_inputs(instance.n, N)))
        self.responses = instance.eval(self.challenges)
        self.N = N
