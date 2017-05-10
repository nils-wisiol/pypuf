from numpy import random, count_nonzero, array, append
import itertools


def random_input(n):
    """
    returns a random {-1,1}-vector of length `n`.
    """
    return random.choice((-1, +1), n)


def all_inputs(n):
    """
    returns an iterator for all {-1,1}-vectors of length `n`.
    """
    return itertools.product((-1, +1), repeat=n)


def random_inputs(n, num):
    """
    returns an iterator for a random sample of {-1,1}-vectors of length `n` (with replacement).
    """
    for i in range(num):
        yield random_input(n)


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


class TrainingSet():

    def __init__(self, instance, N):
        self.challenges = array(list(sample_inputs(instance.n, N)))
        self.responses = instance.eval(self.challenges)
        self.N = N
