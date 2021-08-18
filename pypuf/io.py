from random import sample
from typing import Union, Tuple

import numpy as np
from numpy import int8
from numpy import ndarray
from numpy.random import RandomState

from .simulation import Simulation

BIT_TYPE = int8


def random_inputs(n: int, N: int, seed: int) -> ndarray:
    r"""
    Generates :math:`N` uniformly random challenges of length `n` and returns them as a `numpy.ndarray` of shape
    :math:`(N, n)`. The randomness is based on the provided ``seed``.

    .. note::
        pypuf uses :math:`\{-1,1\}` to represent bit values for both challenges and responses.
        To convert from :math:`\{-1,1\}` notation to the more traditional :math:`\{0,1\}` encoding, use
        ``x = (1 - x) // 2``. For the reverse conversion, use ``x = 1 - 2*x``

        >>> import numpy as np
        >>> import pypuf.io
        >>> challenges = pypuf.io.random_inputs(n=64, N=10, seed=1)
        >>> np.unique(challenges)
        array([-1,  1], dtype=int8)
        >>> challenges01 = (1 - challenges) // 2
        >>> np.unique(challenges01)
        array([0, 1], dtype=int8)
        >>> challenges11 = 1 - 2 * challenges01
        >>> (challenges11 == challenges).all()
        True

    """
    return 2 * RandomState(seed).randint(0, 2, (N, n), dtype=BIT_TYPE) - 1


class ChallengeInformationSet:
    """
    Contains information about the behavior of a PUF token when queried with challenges.
    """
    # TODO fix type annotation when removing python 3.7 support

    def __init__(self, challenges: ndarray, information: ndarray) -> None:
        if challenges.shape[0] != information.shape[0]:
            raise ValueError('Must supply an equal number of challenges and information about these challenges.')
        self.challenges = challenges
        self.information = information
        self.N = len(self.challenges)

    @property
    def challenge_length(self) -> int:
        """
        The length :math:`n` of the recorded challenges.
        """
        return self.challenges.shape[1]

    def __len__(self) -> int:
        return self.challenges.shape[0]

    def __getitem__(self, item: Union[slice, int]) -> Union[Tuple[ndarray, ndarray], object]:
        if isinstance(item, int):
            return self.challenges[item], self.information[item]
        else:
            return self.__class__(self.challenges[item], self.information[item])

    def __eq__(self, other: object) -> bool:
        return (self.challenges == other.challenges).all() and (self.information == other.information).all()

    def random_subset(self, N: Union[int, float]) -> object:
        if N < 1:
            N = int(self.N * N)
        return self[sample(range(self.N), N)]

    def block_subset(self, i: int, total: int) -> object:
        return self[int(i / total * self.N):int((i + 1) / total * self.N)]

    def save(self, f: str) -> None:
        """
        Saves the CRPs to the given file ``f``.
        """
        np.savez(f, challenges=self.challenges, information=self.information)

    @classmethod
    def load(cls, f: str) -> object:
        """
        Loads CRPs from the given file ``f``.
        """
        data = np.load(f)
        return cls(data['challenges'], data['information'])

    def __repr__(self) -> str:
        return f"<{len(self)} CRPs with challenge length {self.challenge_length}>"


class ChallengeResponseSet(ChallengeInformationSet):

    @classmethod
    def from_simulation(cls, instance: Simulation, N: int, seed: int, r: int = 1) -> ChallengeInformationSet:
        challenges = random_inputs(instance.challenge_length, N, seed)
        crp_set = cls(
            challenges=challenges,
            responses=instance.r_eval(r, challenges)
        )
        crp_set.instance = instance
        return crp_set

    def __init__(self, challenges: ndarray, responses: ndarray) -> None:
        """
        Create a challenge-response object containing the given challenges and responses of a PUF token.

        :param challenges: Challenges to the PUF token organized as array of shape :math:`(N, n)`, where :math:`N`
            is the number of challenges and :math:`n` is the challenge length.
        :type challenges: `numpy.ndarray`
        :param responses: Responses of the PUF token organized as array of shape :math:`(N, m, r)`, where :math:`N`
            is the number of challenges, :math:`m` is the response length, and :math:`r` is the number of measurements
            per challenge. The last axis is optional.
        :type responses: `numpy.ndarray`
        """
        super().__init__(challenges, responses)
        if len(responses.shape) == 2:
            responses = responses.reshape(responses.shape + (1, ))
        self.responses = responses

    @property
    def response_length(self) -> int:
        """
        The length :math:`m` of the recorded responses.
        """
        return self.responses.shape[1]

    def __repr__(self) -> str:
        return f"<{len(self)} CRPs with challenge length {self.challenge_length} and response " \
               f"length {self.response_length}, each response measured {self.responses.shape[2]} time(s)>"


class ChallengeReliabilitySet(ChallengeInformationSet):

    @classmethod
    def from_simulation(cls, instance: Simulation, N: int, seed: int, r: int = 5) -> ChallengeInformationSet:
        # noinspection PyTypeChecker
        return cls.from_challenge_response_set(ChallengeResponseSet.from_simulation(instance, N, seed, r))

    @classmethod
    def from_challenge_response_set(cls, crp_set: ChallengeResponseSet) -> ChallengeInformationSet:
        return cls(
            challenges=crp_set.challenges,
            reliabilities=np.average(crp_set.responses, axis=-1),
        )

    def __init__(self, challenges: ndarray, reliabilities: ndarray) -> None:
        super().__init__(challenges, reliabilities)
        self.reliabilities = reliabilities
