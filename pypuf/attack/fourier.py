import numpy as np
from more_itertools import distinct_permutations
from scipy.special import comb

from .base import OfflineAttack
from ..io import ChallengeResponseSet
from ..simulation.base import Simulation

_low_degree_set_cache = {}


def to_low_degree_chisx(X: np.ndarray, deg: int) -> np.ndarray:
    """
    Given a list of inputs `X` of :math:`n` bits each, and the maximum parity function degree `deg`,
    computes the values of all parity functions up to degree `deg` for each of the inputs given in `X`.

    >>> import numpy as np
    >>> X = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]])
    >>> to_low_degree_chisx(X, deg=1)
    array([[ 1,  1,  1,  1,  1],
           [ 1, -1, -1, -1, -1]])

    :param X: List of :math:`N` inputs with :math:`n` bits each, given as numpy array of shape :math:`(N, n)`.
    :param deg: Maximum degree of the parity functions.
    :return: List of :math:`N` vectors containing the values of the parity functions for each given input.
        Represented by a numpy array of shape :math:`(N, k)`, where :math:`k` is the number of parity functions of
        degree up to `deg`.
    """
    return 1 - 2 * (((1 - X) // 2) @ low_degree_sets(X.shape[1], deg).T % 2)


def low_degree_sets(n: int, deg: int = 1) -> np.ndarray:
    """
    Returns a list of parity functions of :math:`n` bit of degree up to `deg`.
    The parity functions are represented by a vector of length :math:`n` indicating of which bits the parity is
    computed.

    >>> low_degree_sets(4, deg=1)
    array([[0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 0],
           [0, 1, 0, 0],
           [1, 0, 0, 0]], dtype=int8)

    Computation can be expensive, results are cached.

    :param n: Number of bits to compute parities from.
    :param deg: Maximum degree of the parity function, i.e. up to how many bits are xored.
    :return: List of parity functions represented by vectors indicating bit positions, formatted as numpy array of
        shape `(num,n)`, where `num` is the number of parity functions.
    """
    if _low_degree_set_cache.get((n, deg)) is not None:
        return _low_degree_set_cache.get((n, deg))

    l = sum(comb(N=n, k=d) for d in range(deg + 1))
    ss = np.empty(shape=(int(l), n), dtype=np.int8)

    idx = 0
    for d in range(deg + 1):
        dl = int(comb(N=n, k=d))
        ss[idx:idx + dl] = np.array(list(distinct_permutations([1] * d + [0] * (n - d), r=n)))
        idx += dl

    _low_degree_set_cache[(n, deg)] = ss
    return ss


class FourierSimulation(Simulation):
    r"""
    A function :math:`f: \{0, 1\}^n \to \mathbb{R}`, defined by its Fourier coefficients.
    """

    def __init__(self, expansion: np.ndarray, deg: int, challenge_length: int, boolean: bool = True) -> None:
        super().__init__()
        self.expansion = expansion
        self.deg = deg
        self._challenge_length = challenge_length
        self.boolean = boolean

    @property
    def challenge_length(self) -> int:
        return self._challenge_length

    @property
    def response_length(self) -> int:
        return self.expansion.shape[0]

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        val = self.eval_chisx(to_low_degree_chisx(challenges, self.deg))
        return np.sign(val) if self.boolean else val

    def eval_chisx(self, chisx: np.ndarray) -> np.ndarray:
        N, f = chisx.shape  # N number of inputs for prediction, f number of Fourier coefficients used for prediction
        m = self.expansion.shape[0]  # number of predictions for each input

        # compute predictions
        Yp = np.empty(shape=(N, m), dtype=np.float32)
        for i in range(m):
            assert self.expansion[i].shape == (f,)
            assert chisx.T.shape == (f, N)
            assert Yp[:, i].shape == (N,)
            Yp[:, i] = self.expansion[i] @ chisx.T
        return Yp


class LMNAttack(OfflineAttack):

    def __init__(self, crps: ChallengeResponseSet, deg: int = 1) -> None:
        r"""
        Given a list of function values of a function :math:`f: \{-1,1\}^n \to \mathbb{R}`, an approximation of the
        Fourier spectrum of the underlying function can be computed.

        The approximation is guaranteed to be correct, if the list contains all function values and the degree equals
        :math:`n`, shown here using the Boolean AND function:

        >>> import numpy as np
        >>> import pypuf.io, pypuf.attack
        >>> challenges = 1 - 2 * np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        >>> responses_and = 1 - 2 * np.array([[0], [0], [0], [1]])
        >>> crps_and = pypuf.io.ChallengeResponseSet(challenges, responses_and)
        >>> (1 - pypuf.attack.LMNAttack(crps_and, 2).fit().eval(challenges)) // 2
        array([[0.],
               [0.],
               [0.],
               [1.]], dtype=float32)

        If additionally the responses are from :math:`\{-1,1\}`, then the sum of squares of the Fourier coefficients
        equals 1, as illustrated here using the majority vote function:

        >>> challenges = 1 - 2 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        >>> responses_maj = 1 - 2 * np.array([[0], [0], [0], [1], [0], [1], [1], [1]])
        >>> crps_maj = pypuf.io.ChallengeResponseSet(challenges, responses_maj)
        >>> exp = pypuf.attack.LMNAttack(crps_maj, 3).fit().expansion
        >>> exp
        array([[ 0. ,  0.5,  0.5,  0.5,  0. ,  0. ,  0. , -0.5]], dtype=float32)
        >>> np.sum(exp**2)
        1.0
        """
        super().__init__(crps)
        self.deg = deg
        self.prediction_shape = None
        self._model = None

    def fit(self) -> FourierSimulation:
        return self.fit_chisx(
            to_low_degree_chisx(self.crps.challenges, self.deg),
            self.crps.responses,
            self.crps.challenge_length,
        )

    def fit_chisx(self, chisx: np.ndarray, Y: np.ndarray, challenge_length: int) -> FourierSimulation:
        # preserve original response shape and flatten after the first axis
        self.prediction_shape = Y.shape[1:]
        Y = Y.reshape((Y.shape[0], -1))

        f = chisx.shape[1]  # number of Fourier coefficients to be approximated
        m = Y.shape[1]  # number of expansions to be computed
        N = Y.shape[0]  # number of examples
        assert N == chisx.shape[0]

        # compute expansions
        expansion = np.empty(shape=(m, f), dtype=np.float32)
        for i in range(m):
            assert Y[:, i].shape == (N,)
            assert chisx.shape == (N, f)
            assert expansion[i].shape == (f,)
            expansion[i] = Y[:, i] @ chisx / f

        # create model
        self._model = FourierSimulation(expansion, self.deg, challenge_length)
        return self.model

    @property
    def model(self) -> Simulation:
        return self._model
