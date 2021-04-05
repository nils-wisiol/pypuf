from itertools import combinations
from typing import List

import numpy as np

from ..io import random_inputs, ChallengeResponseSet
from ..simulation import Simulation


def reliability_data(responses: np.ndarray) -> np.ndarray:
    r"""
    Computes the reliability of response data of a PUF instance.

    Given responses of a PUF instance in an array of shape :math:`(N, m, r)`, where
    :math:`N` is the number of challenges queries,
    :math:`m` is the response length of the PUF,
    :math:`r` is the number of repeated queries for each challenge,
    this function computes the empirical reliability of the PUF responses, i.e. an approximation of

    .. math:: \Pr_\text{eval} \left[ \text{eval}(x) = \text{eval}(x) \right]

    where :math:`\text{eval}(x)` is the response of the (noisy) PUF when evaluated on input :math:`x`, and the
    probability is taken over the noise in the evaluation process.

    An array of shape :math:`(N,m)` is returned, estimating above probability for each challenge and each response bit.
    To obtain a the reliability for each response bit, average along the first axis, to obtain the general reliability,
    average over all axes.

    >>> from pypuf.metrics import reliability_data
    >>> from numpy import average, array
    >>> responses = array([[[1, 1, -1]], [[1, 1, 1]], [[1, 1, 1]], [[1, 1, 1]]])
    >>> average(reliability_data(responses), axis=0)
    array([0.83333333])
    """
    return np.absolute(np.average(responses, axis=-1))


def reliability(instance: Simulation, seed: int, N: int = 10000, r: int = 17) -> np.ndarray:
    r"""
    For a given simulation, estimates simulated reliability.

    Randomly generates :math:`N` challenges using the :math:`\mathtt{seed}`, then queries each challenge :math:`r`
    times, obtaining a response array of shape :math:`(N, m, r)`, where :math:`m` is the response length of the given
    simulation.

    Then applies :func:`reliability_data` to determine the reliability of each respones bit on each challenge.

    Returns a float-array of shape :math:`(N, m)`, giving the reliability of each response bit on each challenge.
    To obtain the general reliability for each response bit, average along the first axis. To obtain the total
    reliability of this instance, average across all axes. Note that bad reliability on single response bits may be
    problematic.

    >>> from pypuf.simulation import XORArbiterPUF
    >>> from pypuf.metrics import reliability
    >>> from numpy import average
    >>> puf = XORArbiterPUF(n=128, k=2, seed=1, noisiness=.2)
    >>> average(reliability(puf, seed=2), axis=0)
    array([0.77729412])
    """
    inputs = random_inputs(n=instance.challenge_length, N=N, seed=seed)
    return np.absolute(reliability_data(instance.r_eval(r, inputs)))


def uniqueness_data(responses: np.ndarray) -> np.ndarray:
    r"""
    Estimates the uniqueness of responses of a set of PUF instances per challenge.

    Given PUF responses in :math:`\{-1,1\}^m` in an array of shape :math:`(l, N, m)`, where
    :math:`l \in \mathbb{N}` is the number of PUF instances,
    :math:`N \in \mathbb{N}` is the number of challenges queried, and
    :math:`m \in \mathbb{N}` is the response length,
    returns an estimate of the uniqueness of PUFs on challenges in this set, i.e. an approximation of

    .. math::
        1 - 2 \cdot E_{f, g; f \neq g} \left[
            \left|
                \frac{1}{2} - \Pr_x \left[ f(x) = g(x) \right]
            \right|
        \right]

    where the challenges :math:`x` are from the (implicitly) provided challenges, :math:`f(x)` and :math:`g(x)` are
    the responses of PUFs :math:`f` and :math:`g` described in the provided data.

    For perfectly unique PUFs and for any :math:`f` and :math:`g`, we expect
    :math:`\Pr_x \left[ f(x) = g(x) \right] = 1/2`, hence the uniqueness to be 1.
    On the other hand, if any two :math:`f` and :math:`g` have similarity higher (or lower) than :math:`1/2`, then
    the uniqueness will be lower than 1.

    Returns an array of shape :math:`(m,)` with uniqueness values per response bit.

    >>> from numpy import random, array
    >>> from pypuf.metrics import uniqueness_data, similarity_data
    >>> prng = random.default_rng(seed=1)
    >>> # generate **different** random responses using numpy's pseudo-random generator
    >>> responses = array([2 * random.default_rng(seed).integers(0, 2, size=(1000, 3)) - 1 for seed in range(4)])
    >>> uniqueness_data(responses)
    array([0.97333333, 0.979     , 0.971     ])
    >>> # generate **same** random responses using numpy's pseudo-random generator
    >>> responses = array([2 * random.default_rng(1).integers(0, 2, size=(1000, 1)) - 1 for seed in range(4)])
    >>> uniqueness_data(responses)
    array([0.])
    """
    if len(responses.shape) == 2:
        # TODO fix LTFArray to return responses of shape (N, 1)
        l, N = responses.shape
        responses = responses.reshape((l, N, 1))

    l, _, m = responses.shape
    similarities = np.empty(shape=(l * (l - 1) // 2, m))
    for idx, (r1, r2) in enumerate(combinations(responses, 2)):
        similarities[idx] = similarity_data(r1, r2)
    res = 1 - 2 * np.average(np.absolute(.5 - similarities), axis=0)
    return res


def uniqueness(instances: List[Simulation], seed: int, N: int = 10000) -> np.ndarray:
    r"""
    Estimates the uniqueness of a list of given Simulations.

    Randomly generates :math:`N` challenges using seed :math:`\texttt{seed}`, then queries each simulation
    each challenge, obtaining an array with shape :math:`(l, N, m)` of all responses, where
    :math:`l` is the number of simulations, and :math:`m` is the response length of the given simulations. (All
    given simulations must have same challenge and response lenght.)

    Then applies :func:`uniqueness_data` to determine the uniqueness of each response bit.

    Returns a float-array of shape :math:`(m,)`, giving the uniqueness of each output bit.
    To obtain total uniqueness, average along all axes, but be aware that low uniqueness on individual response bits
    may be problematic.

    >>> from numpy import average
    >>> from pypuf.simulation import XORArbiterPUF
    >>> from pypuf.metrics import uniqueness
    >>> instances = [XORArbiterPUF(n=64, k=1, seed=seed) for seed in range(5)]
    >>> uniqueness(instances, seed=31415, N=1000)
    array([0.9272])
    """
    m = instances[0].response_length
    challenges = random_inputs(instances[0].challenge_length, N, seed)
    responses = np.empty(shape=(len(instances), N, m))
    for i, instance in enumerate(instances):
        responses[i] = instance.eval(challenges).reshape(N, m)  # TODO fixes incompatible LTFArray output format
    return uniqueness_data(responses)


def similarity_data(responses1: np.ndarray, responses2: np.ndarray) -> np.ndarray:
    r"""
    Given two arrays of responses of shape :math:`(N, m)`, returns the relative frequency of equal responses
    in the arrays for each response bit, i.e. an approximation of

    .. math:: \Pr_x \left[ f(x) = g(x) \right],

    where :math:`f` and :math:`g` are the functions given by :math:`\mathtt{responses1}` and
    :math:`\mathtt{responses2}`, respectively; :math:`x` are the challenges (given implicitly by ordering of the
    response arrays).

    If response length :math:`m` is greater than 1, the similarity is given for each bit seperately. To obtain the
    general similarity, average the response.

    Returns an array of shape :math:`(m,)`.

    >>> from pypuf.metrics import similarity_data
    >>> from numpy import array
    >>> similarity_data(array([[1, 1], [1, 1], [1, 1], [1, 1]]), array([[1, 1], [1, 1], [1, 1], [-1, 1]]))
    array([0.75, 1.  ])
    """
    # Check for response arrays with repetitions (shape (N, m, r))
    # and majority vote along the repetition axis
    if len(responses1.shape) == 3:
        responses1 = np.sign(np.average(responses1, axis=-1))
    if len(responses2.shape) == 3:
        responses2 = np.sign(np.average(responses2, axis=-1))

    # check for broken response arrays with implicit response length 1 (shape (N,))
    # TODO fix LTFArray to return responses of shape (N, 1)
    if len(responses1.shape) == 1:
        responses1 = responses1.reshape(-1, 1)
    if len(responses2.shape) == 1:
        responses2 = responses2.reshape(-1, 1)

    return np.average(np.absolute(responses1 == responses2), axis=0)


def accuracy(simulation: Simulation, test_set: ChallengeResponseSet) -> np.ndarray:
    r"""
    Given a simulation and a test set, returns the relative frequency of responses by the simulation that match
    responses recorded in the test set by response bit, i.e. an approximation of

    .. math:: \Pr_x \left[ f(x) = g(x) \right],

    where :math:`f` is given by the provided simulation and :math:`g` is defined by the :math:`\mathtt{test\_set}`;
    :math:`x` is from the set of challenges given in the :math:`\mathtt{test\_set}`.

    If response length is greater than 1, the similarity is given for each bit seperately. To obtain the
    general similarity, average the response.

    Returns an array of shape :math:`(m,)`.

    >>> from pypuf.simulation import XORArbiterPUF
    >>> from pypuf.io import ChallengeResponseSet
    >>> from pypuf.metrics import accuracy
    >>> puf = XORArbiterPUF(n=128, k=4, noisiness=.1, seed=1)
    >>> test_set = ChallengeResponseSet.from_simulation(puf, N=1000, seed=2)
    >>> accuracy(puf, test_set)
    array([0.823])
    >>> puf = XORArbiterPUF(n=64, k=4, noisiness=.3, seed=2)
    >>> test_set = ChallengeResponseSet.from_simulation(puf, N=1000, seed=2, r=5)
    >>> accuracy(puf, test_set)
    array([0.706])
    """
    sim_responses = simulation.eval(test_set.challenges)
    return similarity_data(sim_responses, test_set.responses)


def similarity(instance1: Simulation, instance2: Simulation, seed: int, N: int = 1000) -> np.ndarray:
    r"""
    Approximate the similarity in response behavior of two simulations with identical challenge and response length,
    i.e. an approximation of

    .. math:: \Pr_x \left[ f(x) = g(x) \right],

    where :math:`f` and :math:`g` are given by the provided simulations, and the :math:`N` challenges :math:`x` are generated
    randomly using the provided :math:`\mathtt{seed}`.

    If response length is greater than 1, the similarity is given for each bit seperately. To obtain the
    general similarity, average the response.

    Returns an array of shape :math:`(m,)`.

    >>> from pypuf.simulation import XORArbiterPUF
    >>> from pypuf.metrics import similarity
    >>> similarity(XORArbiterPUF(n=128, k=4, seed=1), XORArbiterPUF(n=128, k=4, seed=1), seed=31415)  # same seed
    array([1.])
    >>> similarity(XORArbiterPUF(n=128, k=4, seed=1), XORArbiterPUF(n=128, k=4, seed=2), seed=31415)  # different seed
    array([0.515])
    """
    if instance1.challenge_length != instance2.challenge_length:
        raise ValueError(f'Cannot compare instances with different challenge spaces of dimension '
                         f'{instance1.challenge_length} and {instance2.challenge_length}, respectively.')
    inputs = random_inputs(n=instance1.challenge_length, N=N, seed=seed)
    return similarity_data(instance1.eval(inputs), instance2.eval(inputs))
