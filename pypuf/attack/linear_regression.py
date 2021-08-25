from typing import Callable, Optional

import numpy as np

from .base import OfflineAttack
from ..io import ChallengeResponseSet
from ..simulation.base import Simulation


class LinearMapSimulation(Simulation):

    @staticmethod
    def postprocessing_id(responses: np.ndarray) -> np.ndarray:
        return responses

    @staticmethod
    def postprocessing_threshold(responses: np.ndarray) -> np.ndarray:
        return np.sign(responses)

    def __init__(self, linear_map: np.ndarray, challenge_length: int,
                 feature_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 postprocessing: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        super().__init__()
        self.map = linear_map
        self._challenge_length = challenge_length
        self.feature_map = feature_map or (lambda x: x)
        self.postprocessing = postprocessing or self.postprocessing_id

    @property
    def challenge_length(self) -> int:
        return self._challenge_length

    @property
    def response_length(self) -> int:
        return self.map.shape[1]

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        return self.postprocessing(self.feature_map(challenges) @ self.map)


class LeastSquaresRegression(OfflineAttack):

    @staticmethod
    def feature_map_linear(challenges: np.ndarray) -> np.ndarray:
        return challenges

    @staticmethod
    def feature_map_optical_pufs_reloaded(challenges: np.ndarray) -> np.ndarray:
        """
        Computes features of an optical PUF token using all ordered pairs of challenge bits [RHUWDFJ13]_.
        An optical system may be linear in these features.

        .. note::
            This representation is redundant since it treats ordered paris of challenge bits are distinct.
            Actually, only unordered pairs of bits should be treated as distinct. For applications, use
            the function :meth:`feature_map_optical_pufs_reloaded_improved
            <pypuf.attack.LeastSquaresRegression.feature_map_optical_pufs_reloaded_improved>`,
            which achieves the same with half the number of features.

        :param challenges: array of shape :math:`(N, n)` representing challenges to the optical PUF.
        :return: array of shape :math:`(N, n^2)`, which, for each challenge, contains the flattened dyadic product of
            the challenge with itself.
        """
        beta = np.einsum("...i,...j->...ij", challenges, challenges)
        return beta.reshape(beta.shape[:-2] + (-1,))

    @staticmethod
    def feature_map_optical_pufs_reloaded_improved(challenges: np.ndarray) -> np.ndarray:
        r"""
        Computes features of an optical PUF token using all unordered pairs of challenge bits [RHUWDFJ13]_.
        An optical system may be linear in these features.

        :param challenges: 2d array of shape :math:`(N, n)` representing `N` challenges of length :math:`n`.
        :return: array of shape :math:`(N, \frac{n \cdot (n + 1)}{2})`. The result `return[i]` consists of all products
            of unordered pairs taken from `challenges[i]`, which has shape `(N,)`.

        >>> import numpy as np
        >>> import pypuf.attack
        >>> challenges = np.array([[2, 3, 5], [1, 0, 1]])  # non-binary numbers for illustration only.
        >>> pypuf.attack.LeastSquaresRegression.feature_map_optical_pufs_reloaded_improved(challenges)
        array([[ 4,  6, 10,  9, 15, 25],
               [ 1,  0,  1,  0,  0,  1]])
        """
        n = challenges.shape[1]
        idx = np.triu_indices(n)
        return np.einsum("...i,...j->...ij", challenges, challenges)[:, idx[0], idx[1]]

    def __init__(self, crps: ChallengeResponseSet,
                 feature_map: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        super().__init__(crps)
        self.crps = crps
        self.feature_map = feature_map or (lambda x: x)
        self._model = None

    def fit(self) -> Simulation:
        features = self.feature_map(self.crps.challenges)
        # TODO warn if more than one measurement
        linear_map = np.linalg.pinv(features) @ self.crps.responses[:, :, 0]
        self._model = LinearMapSimulation(linear_map, self.crps.challenge_length, self.feature_map)
        return self.model

    @property
    def model(self) -> Simulation:
        return self._model
