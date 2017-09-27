"""
This module provides a set of different functions which can be used e.g. for challenges generation, statistical purpose
or polynomial division. The spectrum is rich and the functions are used in many different modules. Its a kind of a
helper module.
"""
import itertools
from numpy import count_nonzero, array, append, zeros, vstack, mean, prod, ones
from numpy import sum as np_sum
from numpy import abs as np_abs
from numpy.random import RandomState


def random_input(n, random_instance=RandomState()):
    """
    This method generates an array with random interger.
    By default a fresh `numpy.random.RandomState`instance is used.
    :param n: int
              Number of bits which should be generated
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the output bits.
    :returns: array of int
              A pseudo random array of -1 and 1


    """
    return random_instance.choice((-1, +1), n)


def all_inputs(n):
    """
    This functions generates a iterator which produces all possible {-1,1}-vectors.
    :param int
           Length of a n bit vector
    :returns: iterator of {-1,1} int arrays
              An iterator with all possible different {-1,1}-vectors of length `n`.
    """
    return itertools.product((-1, +1), repeat=n)


def random_inputs(n, num, random_instance=RandomState()):
    """
    This function generates an iterator for a random sample of {-1,1}-vectors of length `n` (with replacement).
    If no PRNG provided, a fresh `numpy.random.RandomState` instance is used.
    :param n: int
              Length of a n bit vector
    :param num: int
                Number of n bit vector
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the arrays.
    :return: iterator of num {-1,1} int arrays
             An iterator with num random {-1,1} int arrays.
    """
    for _ in range(num):
        yield random_input(n, random_instance)


def sample_inputs(n, num, random_instance=RandomState()):
    """
    This function generates an iterator for either random samples of {-1,1}-vectors of length `n` if `num` < 2^n,
    and an iterator for all {-1,1}-vectors of length `n` otherwise.
    Note that we return only 2^n vectors even with `num` > 2^n.
    In other words, the output of this function is deterministic if and only if num >= 2^n.
    :param n: int
              Length of a n bit vector
    :param num: int
                Number of n bit vector
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the arrays.
    :return: iterator of num {-1,1} int arrays
             An iterator with num random {-1,1} int arrays depending on num and n.
    """
    return random_inputs(n, num, random_instance) if num < 2 ** n else all_inputs(n)


def iter_append_last(array_iterator, item):
    """
    :param array_iterator: iterator of arbitrary type
                           Iterator where an item should be appended.
    :param item: of arbitrary type equal to iterator type
                 Item which should be appended.
    :returns: iterator of arbitrary type
              An iterator for a given iterator of arrays to which an item will be appended.
    """
    for array_obj in array_iterator:
        yield append(array_obj, item)


def append_last(array_like, item):
    """
    Returns an array for a given array array_like and appends on the axis 1 the element x.
    :param array_like: two dimensional array of type
                       Matrix with initial values
    :param item: type
                 element to be appended
    :return: two dimensional array of type
             initial array_like with appended element item
    """
    append_array = zeros((array_like.shape[0], 1), dtype=int)
    append_array += item
    return append(array_like, append_array, axis=1)


def approx_dist(instance1, instance2, num, random_instance=RandomState()):
    """
    Approximate the distance of two functions instance1, instance2 by evaluating instance1 random set of inputs.
    instance1, instance2 needs to have eval() method and input_length member.
    :param instance1: pypuf.simulation.arbiter_based.base.Simulation
    :param instance2: pypuf.simulation.arbiter_based.base.Simulation
    :param num: int
                Number of n bit vector
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the input arrays.
    :return: float
             Probability (randomly uniform x) for instance1.eval(x) != instance2.eval(x)
    """
    assert instance1.n == instance2.n
    inputs = array(list(random_inputs(instance1.n, num, random_instance)))
    return (num - count_nonzero(instance1.eval(inputs) == instance2.eval(inputs))) / num


def approx_fourier_coefficient(s, training_set):
    """
    Approximate the Fourier coefficient of a function on the subset `s`
    by evaluating the function on `training_set`
    :param s: list of int
                  A {0,1}-array indicating the coefficient's index set
    :param training_set: pypuf.tools.TrainingSet
    :return: float
             The approximated value of the coefficient
    """
    return mean(training_set.responses * chi_vectorized(s, training_set.challenges))


def chi_vectorized(s, inputs):
    """
    Parity function of inputs on indices in s.
    :param s: list of int
              A {0,1}-array indicating the index set
    :param inputs: array of int shape(N,n)
                   {-1,1}-valued inputs to be evaluated.
    :return: array of int shape(N)
             chi_s(x) = prod_(i in s) x_i for all x in inputs (`latex formula`)
    """
    assert len(s) == len(inputs[0])
    result = inputs[:, s > 0]
    if result.size == 0:
        return ones(len(inputs))
    return prod(result, axis=1)


def compare_functions(function1, function2):
    """
    compares two functions on bytecode layer
    :param function1: function object
    :param function2: function object
    :return: bool
    """
    function1_code = function1.__code__
    function2_code = function2.__code__
    functions_equal = function1_code.co_code == function2_code.co_code
    # The bytcode maybe differ from each other https://stackoverflow.com/a/20059029
    functions_equal &= function1_code.co_name == function2_code.co_name
    return functions_equal and function1_code.co_filename == function2_code.co_filename


def transform_challenge_01_to_11(challenge):
    """
    This function is meant to be used with the numpy vectorize method.
    After vectorizing, transform_challenge_01_to_11 can be applied to
    numpy arrays to transform a challenge from 0,1 notation to -1,1 notation.
    :param challenge: array of int
                      Challenge vector in 0,1 notation
    :return: array of int
             Same vector in -1,1 notation
    """
    if (challenge % 2) == 0:
        return 1
    else:
        return -1


def transform_challenge_11_to_01(challenge):
    """
    This function is meant to be used with the numpy vectorize method.
    After vectorizing, transform_challenge_11_to_01 can be applied to
    numpy arrays to transform a challenge from -1,1 notation to 0,1 notation.
    :param challenge: array of int
                      Challenge vector in -1,1 notation
    :return: array of int
             Same vector in 0,1 notation
    """
    if challenge == 1:
        return 0
    else:
        return 1


def poly_mult_div(challenge, irreducible_polynomial, k):
    """
    Return the list of polynomials
        [challenge^2, challenge^3, ..., challenge^(k+1)] mod irreducible_polynomial
    based on the challenge challenge and the irreducible polynomial irreducible_polynomial.
    :param challenge: array of int
                      Challenge vector in 0,1 notation
    :param irreducible_polynomial: array of int
                                   Vector in 0,1 notation
    :param k: int
              Number of PUFs
    :return: array of int
             Array of polynomials
    """
    import polymath as pm

    c_original = challenge
    res = None
    for i in range(k):
        challenge = pm.polymul(challenge, c_original)
        challenge = pm.polymodpad(challenge, irreducible_polynomial)
        if i == 0:
            res = array([challenge])
        else:
            res = vstack((res, challenge))
    return res


def approx_stabilities(instance, num, reps, random_instance=RandomState()):
    """
    This function approximates the stability of the given `instance` for
    `num` challenges evaluating it `reps` times per challenge. The stability
    is the probability that the instance gives the correct response when
    evaluated.
    :param instance: pypuf.simulation.base.Simulation
                     The instance for the stability approximation
    :param num: int
                Amount of challenges to be evaluated
    :param reps: int
                 Amount of repetitions per challenge
    :return: array of float
             Array of the stabilities for each challenge
    """

    challenges = sample_inputs(instance.n, num, random_instance)
    responses = zeros((reps, num))
    for i in range(reps):
        challenges, unpacked_challenges = itertools.tee(challenges)
        responses[i, :] = instance.eval(array(list(unpacked_challenges)))
    return 0.5 + 0.5 * np_abs(np_sum(responses, axis=0)) / reps


class TrainingSet():
    """
    Basic data structure to hold a collection of challenge response pairs.
    Note that this is, strictly speaking, not a set.
    """

    def __init__(self, instance, N):
        """
        :param instance: pypuf.simulation.base.Simulation
                         Instance which is used to generate responses for random challenges.
        :param N: int
                  Number of desired challenges
        """
        self.instance = instance
        self.challenges = array(list(sample_inputs(instance.n, N)))
        self.responses = instance.eval(self.challenges)
        self.N = N
