from __future__ import annotations

from random import sample
from typing import Union

from numpy import abs as np_abs, ndarray, empty
from numpy import copy, int8
from numpy import sum as np_sum
from numpy.random import RandomState

from .simulation.base import Simulation

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


class ChallengeResponseSet:

    def __init__(self, challenges: ndarray, responses: ndarray) -> None:
        if challenges.shape[0] != responses.shape[0]:
            raise ValueError('Must supply an equal number of challenges and responses.')
        self.challenges = challenges
        self.responses = responses
        self.N = len(self.challenges)

    @property
    def challenge_length(self) -> int:
        return self.challenges.shape[1]

    @property
    def response_length(self) -> int:
        return self.responses.shape[1]

    def __len__(self) -> int:
        return self.challenges.shape[0]

    def __getitem__(self, item: Union[slice, int]) -> ChallengeResponseSet:
        return ChallengeResponseSet(self.challenges[item], self.responses[item])

    def random_subset(self, N: Union[int, float]) -> ChallengeResponseSet:
        if N < 1:
            N = int(self.N * N)
        return self[sample(range(self.N), N)]

    def block_subset(self, i: int, total: int) -> ChallengeResponseSet:
        return self[int(i / total * self.N):int((i + 1) / total * self.N)]


class SimulationChallengeResponseSet(ChallengeResponseSet):

    def __init__(self, instance: Simulation, N: int, seed: int, r: int = 1) -> None:
        self.instance = instance
        challenges = random_inputs(instance.challenge_length, N, seed)
        super().__init__(
            challenges=challenges,
            responses=instance.r_eval(r, challenges)
        )
