from ..io import random_inputs
from ..simulation import Simulation
from numpy import ndarray
import numpy as np


def flip_ith_bit(challenges: ndarray, i: int) -> ndarray:
    challenges = challenges[:]
    challenges[:, i] *= -1
    return challenges


def influence(puf: Simulation, i: int, seed: int, N: int = 1000) -> float:
    r"""
    Approximates the influence of the :math:`i`-th input bit on the output of the given PUF simulation.

    For a Boolean function :math:`f:\{-1,1\}^n\to\{-1,1\}`, the influence of the :math:`i`-th input coordinate is
    defined as the probability that the output of the function changes when the :math:`i`-th input coordinate is
    changed, i.e.

    .. math::
        \mathrm{Inf}_i(f) = \Pr_x \left[ f(x) \neq f(x^{\oplus i}) \right],

    where :math:`x^{\oplus i}` is the same as :math:`x`, but with the :math:`i`-th input coordinate flipped.

    The value of :math:`\mathrm{Inf}_i(f)` is approximated on a sample of uniform random inputs to :math:`f`.

    Ideally, all input bits have an influence of approximately 50% on the response of a PUF function.
    In contrast, the Arbiter PUF is known to have input bits with extremely high and low influences:

    >>> import pypuf.simulation, pypuf.metrics
    >>> puf = pypuf.simulation.ArbiterPUF(n=128, seed=1)
    >>> pypuf.metrics.influence(puf, i=0, seed=2)
    0.03
    >>> pypuf.metrics.influence(puf, i=127, seed=3)
    0.971

    The Lightweight Secure PUF [MKP08]_ removed this flaw of the Arbiter PUF, but did not succeed in created a more
    secure PUF [WBMS19]_.

    >>> pypuf.metrics.influence(pypuf.simulation.LightweightSecurePUF(n=128, k=1, seed=1), i=0, seed=2)
    0.447

    :param puf: Function :math:`f`.
    :type puf: :class:`pypuf.simulation.Simulation`
    :param i: The (zero-based) index of the coordinate whose influence will be approximated.
    :type i: ``int``
    :param seed: Determines the seed for the PRNG that generates the uniform random inputs to :math:`f`.
    :type seed: ``int``
    :param N: The number of uniform random inputs that is used to approximate the influence.
    :type N: ``int``
    :return: An approximation of the influence of the :math:`i`-th input coordinate on the output of :math:`f`, i.e.
        :math:`\mathrm{Inf}_i(f)`.
    :rtype: ``float``
    """
    n = puf.challenge_length
    c = random_inputs(n=n, N=N, seed=seed)
    return np.mean(puf.eval(c) != puf.eval(flip_ith_bit(c, i)))


def total_influence(puf: Simulation, seed: int, N: int = 1000) -> float:
    r"""
    The total influence, also known as average sensitivity [ODon14]_, is the sum of all coordinate-wise influences as
    computed by :func:`influence`, i.e.

    .. math::
        \mathrm{I}(f) = \sum_{i=1}^n \Pr_x \left[ f(x) \neq f(x^{\oplus i}) \right].

    We hence have :math:`0 \leq \mathrm{I}(f) \leq n`.

    The total influence can give information about the membership of the function :math:`f` in certain classes and
    is used in PUFMeter to analyze PUFs with respect to modeling attacks [GFS19]_. Low values of total influence are
    concerning as they indicate that a modeling attack using the LMN algorithm may be successful.

    As an example, the total influence of the :class:`pypuf.simulation.ArbiterPUF` and
    :class:`pypuf.simulation.PermutationPUF` can be computed as follows:

    >>> import pypuf.simulation, pypuf.metrics
    >>> pypuf.metrics.total_influence(pypuf.simulation.ArbiterPUF(n=64, seed=1), seed=2)
    19.358
    >>> pypuf.metrics.total_influence(pypuf.simulation.PermutationPUF(n=64, k=1, seed=1), seed=2)
    32.263...

    :param puf: Function :math:`f`.
    :type puf: :class:`pypuf.simulation.Simulation`
    :param seed: Determines the seed for the PRNG that generates the uniform random inputs to :math:`f`.
    :type seed: ``int``
    :param N: The number of uniform random inputs that is used to approximate the influence.
    :type N: ``int``
    :return: An approximation of the total influence of :math:`f`, i.e. :math:`\mathrm{I}_i(f)`.
    :rtype: ``float``
    """
    n = puf.challenge_length
    c = random_inputs(n=n, N=N, seed=seed)
    r = puf.eval(c)  # original responses
    return sum(
        np.mean(r != puf.eval(flip_ith_bit(c, i)))
        for i in range(n)
    )


def noise_sensitivity(puf: Simulation, eps: float, seed: int, N: int = 1000) -> float:
    r"""
    The noise sensitivity captures how a function reacts to noisy inputs. Note that noise here refers to flipped
    *input bits*, i.e. is different from the usual notion of noise in the context of PUFs. Formally, the noise
    sensitivity of a Boolean function :math:`f` at noise-level :math:`\varepsilon` is defined as the probability that
    the output flips if each input bit is flipped with probability :math:`\varepsilon` (independently). I.e.,

    .. math::
        \mathrm{NS}_\varepsilon(f) = \Pr_x \left[ f(x) \neq f(x') \right],

    where :math:`x'` is a copy of :math:`x`, but each bit is flipped independently with probability :math:`\varepsilon`.

    The noise sensitivity is approximated using a uniform random sample of inputs and corresponding
    with-probability-:math:`\varepsilon`-flipped inputs to :math:`f`.

    A low noise sensitivity of :math:`f` indicates that further testing is required, as :math:`f` may be close to a
    junta and thus susceptible to a modeling attack [GFS19]_.

    Usually, it is interesting to compute the noise sensitivity for small :math:`\varepsilon`:

    >>> import pypuf.simulation, pypuf.metrics
    >>> puf = pypuf.simulation.ArbiterPUF(n=64, seed=1)
    >>> pypuf.metrics.noise_sensitivity(puf, eps=.01, seed=2)
    0.216

    :param puf: Function :math:`f`.
    :type puf: :class:`pypuf.simulation.Simulation`
    :param eps: Noise-level :math:`\varepsilon` of the inputs.
    :type eps: ``float``
    :param seed: Determines the seed for the PRNG that generates the inputs to :math:`f`.
    :type seed: ``int``
    :param N: The number of uniform random inputs that is used to approximate the noise sensitivity.
    :type N: ``int``
    :return: The noise sensitivity of :math:`f` at noise level :math:`\varepsilon`, i.e.
        :math:`\mathrm{NS}_\varepsilon(f)`.
    :rtype: ``float``
    """
    n = puf.challenge_length
    challenges = random_inputs(n=n, N=N, seed=seed)
    flip = np.random.default_rng(seed).choice([-1, 1], size=(N, n), p=[eps, 1 - eps])
    return np.mean(puf.eval(challenges) != puf.eval(challenges * flip))
