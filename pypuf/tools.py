from numpy import count_nonzero, array, append, zeros, vstack, mean, prod, ones, sum, abs
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


def sample_inputs(n, num, random_instance=RandomState()):
    """
    returns an iterator for either random samples of {-1,1}-vectors of length `n` if `num` < 2^n,
    and an iterator for all {-1,1}-vectors of length `n` otherwise.
    Note that we return only 2^n vectors even with `num` > 2^n.
    In other words, the output of this function is deterministic if and only if num >= 2^n.
    """
    return random_inputs(n, num, random_instance) if num < 2 ** n else all_inputs(n)


def iter_append_last(array_iterator, x):
    """
    returns an iterator for a given iterator of arrays to which element x will be appended
    """
    for array in array_iterator:
        yield append(array, x)


def approx_dist(a, b, num, random_instance=RandomState()):
    """
    Approximate the distance of two functions a, b by evaluating a random set of inputs.
    a, b needs to have eval() method and input_length member.
    :return: probability (randomly uniform x) for a.eval(x) != b.eval(x)
    """
    assert a.n == b.n
    d = 0
    inputs = array(list(random_inputs(a.n, num, random_instance)))
    return (num - count_nonzero(a.eval(inputs) == b.eval(inputs))) / num


def approx_fourier_coefficient(s, training_set):
    """
    Approximate the Fourier coefficient of a function on the subset `s`
    by evaluating the function on `training_set`
    """
    return mean(training_set.responses * chi_vectorized(s, training_set.challenges))


def chi_vectorized(s, inputs):
    """
    :return: chi_s(x) = prod_(i \in s) x_i for all x in inputs
    """
    assert len(s) == len(inputs[0])
    result = inputs[:, s > 0]
    if result.size == 0:
        return ones(len(inputs))
    return prod(result, axis=1)


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


def transform_challenge_01_to_11(a):
    """
    This function is meant to be used with the numpy vectorize method.
    After vectorizing, transform_challenge_01_to_11 can be applied to
    numpy arrays to transform a challenge from 0,1 notation to -1,1 notation.
    :param a: challenge vector in 0,1 notation
    :return: same vector in -1,1 notation
    """
    if (a % 2) == 0:
        return 1
    else:
        return -1


def transform_challenge_11_to_01(a):
    """
    This function is meant to be used with the numpy vectorize method.
    After vectorizing, transform_challenge_11_to_01 can be applied to
    numpy arrays to transform a challenge from -1,1 notation to 0,1 notation.
    :param a: challenge vector in -1,1 notation
    :return: same vector in 0,1 notation
    """
    if a == 1:
        return 0
    else:
        return 1


def poly_mult_div(c, f, k):
    """
    Return the list of polynomials
        [c^2, c^3, ..., c^(k+1)] mod f
    based on the challenge c and the irreducible polynomial f.
    """
    import polymath as pm

    c_original = c
    global res
    for i in range(k):
        c = pm.polymul(c, c_original)
        c = pm.polymodpad(c, f)
        if i == 0:
            res = array([c])
        else:
            res = vstack((res, c))
    return res


def approx_stabilities(instance, num, reps, random_instance=RandomState()):
    """
        This function approximates the stability of the given `instance` for
        `num` challenges evaluating it `reps` times per challenge. The stability
        is the probability that the instance gives the correct response when
        evaluated.
        :param instance: The instance for the stability approximation
        :param num: Amount of challenges to be evaluated
        :param reps: Amount of repetitions per challenge
        :return: list of the stabilities for each challenge
    """

    challenges = sample_inputs(instance.n, num, random_instance)
    responses = zeros((reps, num))
    for i in range(reps):
        challenges, cs = itertools.tee(challenges)
        responses[i, :] = instance.eval(array(list(cs)))
    return 0.5 + 0.5 * abs(sum(responses, axis=0)) / reps


class TrainingSet():
    """
    Basic data structure to hold a collection of challenge response pairs.
    Note that this is, strictly speaking, not a set.
    """

    def __init__(self, instance, N):
        self.instance = instance
        n = instance.n
        if instance.bias:
            n = n - 1
        self.challenges = array(list(sample_inputs(n, N)))
        self.responses = instance.eval(self.challenges)
        self.N = N
