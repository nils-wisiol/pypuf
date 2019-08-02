"""
This module provides a set of different functions which can be used e.g. for challenges generation, statistical purpose
or polynomial division. The spectrum is rich and the functions are used in many different modules. Its a kind of a
helper module.
"""
import itertools
from importlib import import_module
from inspect import getmembers, isclass
from math import ceil, log
from random import sample

from numpy import abs as np_abs
from numpy import absolute
from numpy import copy as np_copy
from numpy import count_nonzero, array, append, zeros, vstack, mean, prod, ones, dtype, full, shape, copy, int8, \
    multiply, empty, average
from numpy import squeeze
from numpy import sum as np_sum
from numpy.random import RandomState

from pypuf.simulation.base import Simulation
from pypuf.studies.base import Study

BIT_TYPE = int8


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
    return random_inputs(n, 1, random_instance)[0]


def all_inputs(n):
    """
    This functions generates a iterator which produces all possible {-1,1}-vectors.
    :param int
           Length of a n bit vector
    :returns: array of int8
              An array with all possible different {-1,1}-vectors of length `n`.
    """
    return (array(list(itertools.product((-1, +1), repeat=n)))).astype(BIT_TYPE)


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
    return 2 * random_instance.randint(0, 2, (num, n), dtype=BIT_TYPE) - 1


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
    assert arr.dtype == dtype(type(item)), 'The elements of arr and item must be of the same type, but the array has ' \
                                           'type %s and the item has type %s.' % (arr.dtype, dtype(type(item)))
    dimension = list(shape(arr))
    assert len(dimension) >= 1, 'arr must have at least one dimension.'
    # the lowest level should contain one item
    dimension[-1] = 1
    # create an array white shape(array) where the lowest level contains only one item
    item_arr = full(dimension, item, dtype=BIT_TYPE)
    # the item should be appended at the lowest level
    axis = len(dimension) - 1
    return append(arr, item_arr, axis=axis)


def approx_dist(instance1: Simulation, instance2: Simulation, num, random_instance=RandomState()):
    """
    Approximate the distance of two Simulations instance1, instance2 by evaluating a random set of inputs.
    The image of instance1 and instance2 needs to be {-1,1}, and they must have identical challenge_length().
    :param instance1: pypuf.simulation.arbiter_based.base.Simulation
    :param instance2: pypuf.simulation.arbiter_based.base.Simulation
    :param num: int
                Number of n bit vector
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the input arrays.
    :return: float
             Probability (randomly uniform x) for instance1.eval(x) != instance2.eval(x)
    """
    assert instance1.challenge_length() == instance2.challenge_length(), \
        'Cannot compare instances with different challenge spaces of dimension %i and %i, respectively.' \
        % (instance1.challenge_length(), instance2.challenge_length())
    inputs = random_inputs(instance1.challenge_length(), num, random_instance=random_instance)
    return (num - count_nonzero(instance1.eval(inputs) == instance2.eval(inputs))) / num


def approx_dist_real(instance1: Simulation, instance2: Simulation, num, random_instance=RandomState()):
    """
    Approximate the distance of two Simulations instance1, instance2 by evaluating a random set of inputs.
    Both must have identical challenge_length().
    :param instance1: pypuf.simulation.arbiter_based.base.Simulation
    :param instance2: pypuf.simulation.arbiter_based.base.Simulation
    :param num: int
                Number of n bit vector
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the input arrays.
    :return: float
             Generalization of approx_dist(instance1, instance2, ...),
             Expectation (randomly uniform x) |instance1.eval(x) - instance2.eval(x)| / 2
    """
    assert instance1.challenge_length() == instance2.challenge_length(), \
        'Cannot compare instances with different challenge spaces of dimension %i and %i, respectively.' \
        % (instance1.challenge_length(), instance2.challenge_length())
    inputs = random_inputs(instance1.challenge_length(), num, random_instance=random_instance)
    return average(absolute(instance1.eval(inputs) - instance2.eval(inputs)))


def approx_dist_nonrandom(instance, test_set):
    """
    Approximate the distance of function instance to the hypothetical function
    that generated the challenge-response pairs in test_set.
    :param instance: pypuf.simulation.arbiter_based.base.Simulation
                     Model to evaluate
    :param test_set: pypuf.tools.TrainingSet
                     Challenge-response pairs to test instance with
    :return: float
             Ratio of correctly to incorrectly predicted responses
    """
    return (test_set.N - count_nonzero(instance.eval(test_set.challenges) == test_set.responses)) / test_set.N


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
    assert len(s) == len(inputs[0])
    result = inputs[:, s > 0]
    if result.size == 0:
        return ones(len(inputs), dtype=BIT_TYPE)
    return prod(result, axis=1, dtype=BIT_TYPE)


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
    res = np_copy(challenge)
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
    res = np_copy(challenge)
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
    # TODO Change the type to int8 or uint8
    challenge = challenge.astype('uint8')
    irreducible_polynomial = irreducible_polynomial.astype('uint8')
    # TODO Change the type to int8 or uint8
    # challenge = challenge.astype('int64')
    # irreducible_polynomial = irreducible_polynomial.astype('int64')
    c_original = challenge
    res = None
    for i in range(k):
        challenge = pm.polymul(challenge, c_original)
        challenge = pm.polymodpad(challenge, irreducible_polynomial)
        if i == 0:
            res = array([challenge], dtype='int8')
        else:
            res = vstack((res, challenge))
    res = res.astype(BIT_TYPE)
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
    This function checks the type of the array to match the BIT_TYPE
    :param arr: array of arbitrary type
    """
    assert arr.dtype == dtype(BIT_TYPE), 'Must be an array of {0}. Got array of {1}'.format(BIT_TYPE, arr.dtype)


def parse_file(filename, n, start=1, num=0, in_11_notation=False):
    """
    Reads challenge-response pairs from a file.
    The format is one pair per line, first all n inputs (challenge) separated
    by spaces followed by a single output (response) value.
    :param filename: string
                     Path of the file to read the challenge-response pairs from
    :param n: int
              Challenge bits
    :param start: int
                  First line to read
    :param num: int
                Number of lines to read, 0 to read the whole file
    :param in_11_notation: bool
                           Format the file is in
                           True for -1,1 notation, False for 0,1
    :return: tools.TrainingSet
             A TraningSet with the num challenges and responses that were read
    """
    if num == 0:
        stop = float('inf')
    else:
        stop = start + num
    if in_11_notation:
        allowed_vals = ['-1', '1']
    else:
        allowed_vals = ['0', '1']

    challenges, responses = [], []
    with open(filename) as f:
        for ln, line in enumerate(f):
            if start <= ln + 1 < stop:
                vals = line.split()
                assert len(vals) == n + 1, \
                    'Line {} contains {} values, expected {}' \
                    .format(ln + 1, len(vals), n + 1)
                assert set(vals).issubset(allowed_vals), \
                    'Line {} contains an invalid value: {}' \
                    .format(ln + 1, next(iter(set(vals).difference(allowed_vals))))
                challenges.append(vals[:n])
                responses.append(vals[n])

    if num == 0:
        num = len(challenges)
    assert len(challenges) == num, \
        'File contains insufficient lines ({} read, {} needed)' \
        .format(len(challenges), num)

    challenges = array(challenges).astype(BIT_TYPE)
    responses = array(responses).astype(BIT_TYPE)

    if not in_11_notation:
        challenges = transform_challenge_01_to_11(challenges)
        responses = transform_challenge_01_to_11(responses)

    return ChallengeResponseSet(challenges, responses)


class ChallengeResponseSet:
    """
    A set of challenges and corresponding responses.
    """

    def __init__(self, challenges, responses):
        """
        Create a set of challenges and corresponding responses. Note that the order of the
        challenges and responses parameter is relevant.
        :param challenges: List of challenges
        :param responses: List of responses, ordered accordingly
        """
        self.challenges = challenges
        self.responses = responses
        assert len(self.challenges) == len(self.responses)
        self.N = len(self.challenges)

    def random_subset(self, N):
        """
        Gives a random subset of this challenge response set.
        :param N: Either a relative (to the total number) or absolute number of challenges.
        :return: A random subset samples from this challenge response set.
        """
        if N < 1:
            N = int(self.N * N)
        return self.subset(sample(range(self.N), N))

    def block_subset(self, i, total):
        """
        Gives the i-th block of this challenge response set.
        :param i: Index of the block that is to be returned.
        :param total: Total number of blocks.
        :return: A challenge response set.
        """
        return self.subset(slice(
            int(i / total * self.N),
            int((i + 1) / total * self.N)
        ))

    def subset(self, subset_slice):
        """
        Gives the subset of this challenge response set defined by the slice given.
        :param subset_slice: A python array slice
        :return: A challenge response set defined accordingly
        """
        return ChallengeResponseSet(
            challenges=self.challenges[subset_slice],
            responses=self.responses[subset_slice]
        )


class TrainingSet(ChallengeResponseSet):
    """
    Basic data structure to hold a collection of challenge response pairs.
    Note that this is, strictly speaking, not a set.
    """

    def __init__(self, instance, N, random_instance=RandomState(), reps=1):
        """
        :param instance:        pypuf.simulation.base.Simulation
                                Instance which is used to generate responses for random challenges
        :param N:               int
                                Number of desired challenges
        :param random_instance: numpy.random.RandomState
                                PRNG instance for pseudo random sampling challenges
        :param reps:            int
                                Number of repeated evaluations of every challenge on instance (None equals 1)
        """
        self.instance = instance
<<<<<<< HEAD
        challenges = sample_inputs(instance.n, N, random_instance=random_instance)
=======
        self.N = min(N, 2 ** instance.n)
        challenges = sample_inputs(instance.n, N, random_instance=random_instance)
        responses = zeros((reps, self.N))
        for i in range(reps):
            challenges, copy = itertools.tee(challenges)
            responses[i, :] = instance.eval(array(list(copy)))
        if reps == 1:
            responses = squeeze(responses, axis=0)
        self.reps = reps
>>>>>>> 03e3107... all in one commit
        super().__init__(
            challenges=challenges,
            responses=responses,
        )


class GoldreichLevin:
    """
    Probabilistic algorithm that with probability 1 - `delta` returns a list of sets for the `instance` Boolean function
    using query access.
    If the magnitude of a coefficient is greater or equal than `tau` its set is guaranteed to be in the output list.
    On the other hand all sets in the output list are guaranteed to have Fourier coefficient magnitude greater or equal
    than 1/2 `tau`.
    """

    def __init__(self, instance: Simulation, tau, delta):
        self.instance = instance
        self.tau = tau
        epsilon = tau ** 2 / 4
        self.delta = tau ** 2 / (8 * self.instance.challenge_length() * (1 - delta))
        self.sample_size = int(ceil(12 * log(2.0 / self.delta) / (epsilon ** 2)))

    def find_heavy_monomials(self, logger=None):
        """
        Returns a list of monomials on which the Fourier weight is concentrated.
        If the Fourier weight is not concentrated on a small set, this will have
        LONG runtime.
        :param logger: if given, used to log status messages
        """
        return self._recursive_find((0, zeros(self.instance.challenge_length())), logger)

    def _recursive_find(self, bucket, logger=None):
        k = bucket[0]
        s = bucket[1]
        if k == self.instance.challenge_length():
            return [s]

        extended_s = copy(s)
        extended_s[k] = 1
        next_buckets = [(k + 1, s), (k + 1, extended_s)]

        return_sets = []
        for new_bucket in next_buckets:
            weight = self._sample_weight(new_bucket)
            if weight > self.tau ** 2 / 2:
                if logger:
                    logger.debug(f'disecting bucket with weight {weight} > {self.tau ** 2 / 2}')
                    logger.debug(f'bucket: {new_bucket[1]}')
                return_sets += self._recursive_find(new_bucket, logger)

        return return_sets

    def _sample_weight(self, bucket):
        k = bucket[0]
        s = bucket[1]
        j = array([1 if i < k else 0 for i in range(self.instance.challenge_length())])

        z = random_inputs(self.instance.challenge_length() - k, self.sample_size)
        x1 = append(random_inputs(k, self.sample_size), z, axis=1)
        x2 = append(random_inputs(k, self.sample_size), z, axis=1)
        mask = multiply(s, j)
        estimate = average(
            self.instance.eval(x1) * self.chi(mask, x1) *
            self.instance.eval(x2) * self.chi(mask, x2)
        )

        return estimate

    @staticmethod
    def chi(s, x):
        """
        For each y in x, returns chi_s(y) = prod_(i ∈ s) y_i.

        >>> GoldreichLevin.chi(array([0,1,1]), array([[-1,-1,-1]]))
        array([1.])

        >>> GoldreichLevin.chi(array([0, 1]), array([[1,1],[-1,-1],[-1,1],[1,-1]]))
        array([ 1., -1.,  1., -1.])
        """
        masked_values = empty(shape=x.shape)
        for i, bit in enumerate(s):
            masked_values[:, i] = x[:, i] if bit == 1 else ones(x.shape[0])
        return prod(masked_values, axis=1)


def find_study_class(name):
    """
    Returns the study class object for a given module name. The name can include the 'pypuf.studies'
    prefix, but does not have to. The returned class is guaranteed to be a descendant to pypuf.studies.base.Study.
    If the given module does not contain exactly one Study class, an error will be PRINTED and python will EXIT.
    :param name: Module name with our without prefix, containting exactly one descendant to Study.
    :return: Study class
    """
    if not name.startswith('pypuf.studies'):
        name = 'pypuf.studies.' + name

    try:
        study_module = import_module(name)
    except ModuleNotFoundError as e:
        print('Module {} cannot be imported due to the following exception:'.format(name))
        print(e)
        exit(1)

    studies = [
        c[1] for c in getmembers(study_module, isclass)
        if isinstance(c, tuple) and
        len(c) > 1 and
        str(c[1].__module__).startswith(name) and
        issubclass(c[1], Study)
    ]

    if not studies:
        print('Module {} does not contain any study.'.format(name))
        exit(1)

    if len(studies) > 1:
        print('Module {} contains more than one study:'.format(name))
        for s in studies:
            print(' - {}'.format(s))
        exit(1)

    return studies[0]
