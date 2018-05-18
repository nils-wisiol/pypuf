"""
This module provides a set of different functions which can be used e.g. for challenges generation, statistical purpose
or polynomial division. The spectrum is rich and the functions are used in many different modules. Its a kind of a
helper module.
"""
import itertools
from numpy import count_nonzero, array, append, zeros, vstack, mean, prod, ones, dtype, full, shape, copy, fromstring
from numpy import sum as np_sum
from numpy import abs as np_abs
from numpy.random import RandomState

RESULT_TYPE = 'int8'


def random_input(n, random_instance=RandomState()):
    """
    This method generates an array with random integer.
    By default a fresh `numpy.random.RandomState`instance is used.
    :param n: int
              Number of bits which should be generated
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the output bits.
    :returns: array of int8
              A pseudo random array of -1 and 1
    """
    return (random_instance.choice((-1, +1), n)).astype(RESULT_TYPE)


def all_inputs(n):
    """
    This functions generates a iterator which produces all possible {-1,1}-vectors.
    :param int
           Length of a n bit vector
    :returns: array of int8
              An array with all possible different {-1,1}-vectors of length `n`.
    """
    return (array(list(itertools.product((-1, +1), repeat=n)))).astype(RESULT_TYPE)


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
    :return: array of num {-1,1} int8 arrays
             An array with num random {-1,1} int arrays.
    """
    res = zeros((num, n), dtype=RESULT_TYPE)
    for i in range(num):
        res[i] = random_input(n, random_instance=random_instance)
    return res


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
    :return: array of num {-1,1} int8 arrays
             An array with num random {-1,1} int arrays depending on num and n.
    """
    return random_inputs(n, num, random_instance) if num < 2 ** n else all_inputs(n)


def append_last(arr, item):
    """
    Returns an array for a given array arr and appends on the lowest level the element item.
    :param arr: n dimensional array of type
                       Matrix with initial values
    :param item: type
                 element to be appended
    :return: n dimensional array of type
             initial arr with appended element item
    """
    dimension = list(shape(arr))
    assert len(dimension) >= 1, 'arr must have at least one dimension.'
    # the lowest level should contain one item
    dimension[-1] = 1
    # create an array white shape(array) where the lowest level contains only one item
    item_arr = full(dimension, item)
    # the item should be appended at the lowest level
    axis = len(dimension) - 1
    return append(arr, item_arr, axis=axis)


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
    inputs = random_inputs(instance1.n, num, random_instance=random_instance)
    return (num - count_nonzero(instance1.eval(inputs) == instance2.eval(inputs))) / num


def approx_fourier_coefficient(s, training_set):
    """
    Approximate the Fourier coefficient of a function on the subset `s`
    by evaluating the function on `training_set`
    :param s: list of int8
                  A {0,1}-array indicating the coefficient's index set
    :param training_set: pypuf.tools.TrainingSet
    :return: float
             The approximated value of the coefficient
    """
    assert_result_type(s)
    assert_result_type(training_set.challenges)
    return mean(training_set.responses * chi_vectorized(s, training_set.challenges))


def chi_vectorized(s, inputs):
    """
    Parity function of inputs on indices in s.
    :param s: list of int8
              A {0,1}-array indicating the index set
    :param inputs: array of int8 shape(N,n)
                   {-1,1}-valued inputs to be evaluated.
    :return: array of int8 shape(N)
             chi_s(x) = prod_(i in s) x_i for all x in inputs (`latex formula`)
    """
    assert_result_type(s)
    assert_result_type(inputs)
    assert len(s) == len(inputs[0])
    result = inputs[:, s > 0]
    if result.size == 0:
        return ones(len(inputs), dtype=RESULT_TYPE)
    return prod(result, axis=1, dtype=RESULT_TYPE)


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
    This function is used to transform a challenge from 0,1 notation to -1,1 notation.
    :param challenge: array of int8
                      Challenge vector in 0,1 notation
    :return: array of int8
             Same vector in -1,1 notation
    """
    assert_result_type(challenge)
    res = copy(challenge)
    res[res == 1] = -1
    res[res == 0] = 1
    return res


def transform_challenge_11_to_01(challenge):
    """
    This function is used to transform a challenge from -1,1 notation to 0,1 notation.
    :param challenge: array of int8
                      Challenge vector in -1,1 notation
    :return: array of int8
             Same vector in 0,1 notation
    """
    assert_result_type(challenge)
    res = copy(challenge)
    res[res == 1] = 0
    res[res == -1] = 1
    return res


def poly_mult_div(challenge, irreducible_polynomial, k):
    """
    Return the list of polynomials
        [challenge^2, challenge^3, ..., challenge^(k+1)] mod irreducible_polynomial
    based on the challenge challenge and the irreducible polynomial irreducible_polynomial.
    :param challenge: array of int8
                      Challenge vector in 0,1 notation
    :param irreducible_polynomial: array of int8
                                   Vector in 0,1 notation
    :param k: int
              Number of PUFs
    :return: array of int8
             Array of polynomials
    """
    import polymath as pm
    assert_result_type(challenge)
    assert_result_type(irreducible_polynomial)
    c_original = challenge
    res = None
    for i in range(k):
        challenge = pm.polymul(challenge, c_original)
        challenge = pm.polymodpad(challenge, irreducible_polynomial)
        if i == 0:
            res = array([challenge], dtype=RESULT_TYPE)
        else:
            res = vstack((res, challenge))
    res = res.astype(RESULT_TYPE)
    assert_result_type(res)
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
        responses[i, :] = instance.eval(challenges)
    return 0.5 + 0.5 * np_abs(np_sum(responses, axis=0)) / reps


def assert_result_type(arr):
    """
    This function checks the type of the array to match the RESULT_TYPE
    :param arr: array of arbitrary type
    """
    assert arr.dtype == dtype(RESULT_TYPE), 'Must be an array of {0}. Got array of {1}'.format(RESULT_TYPE, arr.dtype)


def crps_from_file(filename):
    """
    This function reads challenge response pairs from a file and stores them into a tuple. If a challenge occurs
    multiple times majority vote determines the response. All challenges should have the same dimension n.
    :param filename: string
                     A path to a file with the format:
                     challenge,response\n
                     ...
                     challenge,response\n
                     The challenge is written in hex and the response is '0' or '1'.
    :return: Tuple of challenge and responses. The shape of the arrays depends on the number of different challenges and
             the challenge dimension.
             (array(array of RESULT_TYPE), array of RESULT_TYPE)
    """
    crp_dict = dict()
    challenges = []
    responses = []
    with open(filename, 'r') as file:
        for line in file:
            crp_list = line.strip().split(',')
            crp_dict.setdefault(crp_list[0], [])
            crp_dict[crp_list[0]].append(crp_list[1])

    for key, value in crp_dict.items():
        # one hex digit results in 4 binary digits
        digits = len(key) * 4
        # converts the hex string into a whitespace separated bit string
        str_chl = ' '.join(bin(int(key, base=16)).lstrip('0b').zfill(digits))
        # converts the string challenge into  a numpy array
        chl_arry_01 = fromstring(str_chl, dtype=RESULT_TYPE, sep=' ')
        # transform challenge into -1,1 notation and safe it
        challenges.append(transform_challenge_01_to_11(chl_arry_01))
        one_count = value.count('1')
        zero_count = value.count('0')
        assert one_count != zero_count, 'The number of 0 and 1 should not be equal. Use a odd number of responses.'
        # majority vote responses
        if one_count > zero_count:
            responses.append(-1)
        else:
            responses.append(1)
    return (array(challenges), array(responses, dtype=RESULT_TYPE))

class TrainingSet():
    """
    Basic data structure to hold a collection of challenge response pairs.
    Note that this is, strictly speaking, not a set.
    """

    def __init__(self, instance, N, random_instance=RandomState()):
        """
        :param instance: pypuf.simulation.base.Simulation
                         Instance which is used to generate responses for random challenges.
        :param N: int
                  Number of desired challenges
        :param random_instance: numpy.random.RandomState
                                PRNG which is used to draft challenges.
        """
        self.instance = instance
        self.challenges = sample_inputs(instance.n, N, random_instance=random_instance)
        self.responses = instance.eval(self.challenges)
        self.N = N
