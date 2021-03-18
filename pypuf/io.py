from random import sample
from typing import Union

from numpy import abs as np_abs, ndarray, empty
from numpy import copy, int8
from numpy import sum as np_sum
from numpy.random import RandomState
import numpy as np

from .simulation import Simulation

BIT_TYPE = int8


# TODO add documentation


def random_inputs(n: int, N: int, seed: int) -> ndarray:
    return 2 * RandomState(seed).randint(0, 2, (N, n), dtype=BIT_TYPE) - 1


def transform_challenge_01_to_11(challenge: ndarray) -> ndarray:  # TODO remove
    res = copy(challenge)
    res[res == 1] = -1
    res[res == 0] = 1
    return res


def transform_challenge_11_to_01(challenge: ndarray) -> ndarray:  # TODO remove
    res = copy(challenge)
    res[res == 1] = 0
    res[res == -1] = 1
    return res


def approx_stabilities(instance: Simulation, num: int, reps: int, seed: int) -> ndarray:
    challenges = random_inputs(instance.challenge_length, num, seed)
    responses = empty((reps, num), dtype=BIT_TYPE)
    for i in range(reps):
        responses[i, :] = instance.eval(challenges)
    return 0.5 + 0.5 * np_abs(np_sum(responses, axis=0)) / reps


class ChallengeInformationSet:
    # TODO fix type annotation when removing python 3.7 support

    def __init__(self, challenges: ndarray, information: ndarray) -> None:
        if challenges.shape[0] != information.shape[0]:
            raise ValueError('Must supply an equal number of challenges and information about these challenges.')
        self.challenges = challenges
        self.information = information
        self.N = len(self.challenges)

    @property
    def challenge_length(self) -> int:
        return self.challenges.shape[1]

    def __len__(self) -> int:
        return self.challenges.shape[0]

    def __getitem__(self, item: Union[slice, int]) -> object:
        return self.__class__(self.challenges[item], self.information[item])

    def random_subset(self, N: Union[int, float]) -> object:
        if N < 1:
            N = int(self.N * N)
        return self[sample(range(self.N), N)]

    def block_subset(self, i: int, total: int) -> object:
        return self[int(i / total * self.N):int((i + 1) / total * self.N)]


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
        super().__init__(challenges, responses)
        self.responses = responses

    @property
    def response_length(self) -> int:
        return self.responses.shape[1]


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
