from os import getpid
from typing import NamedTuple
from uuid import UUID

from numpy import broadcast_to, average, absolute, ones
from numpy.core.records import ndarray
from numpy.random.mtrand import RandomState
from seaborn import catplot

from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF
from pypuf.simulation.base import Simulation
from pypuf.studies.base import Study
from pypuf.tools import random_inputs


class Parameters(NamedTuple):
    n: int
    r: int
    c: int
    c1: int
    p: int
    seed: int
    noisiness: float
    transform: str


class Result(NamedTuple):
    experiment_id: UUID
    pid: int
    measured_time: float
    reliability: float


class Sponge(Simulation):

    def __init__(self, n: int, r: int, c: int, c1: int, p: int, transform: str, seed: int, noisiness: float) -> None:
        super().__init__()
        assert r * p == n, f'r*p must be n, but had r={r}, p={p}, r*p={r*p}!={n}=n.'
        self.n = n
        self.r = r
        self.c = c
        self.c1 = c1
        self.p = p
        self.f = RandomState(0).permutation(n)
        self.pufs = [
            XORArbiterPUF(n=c, k=1, seed=i + c * seed, transform=transform,
                          noisiness=noisiness, noise_seed=i + c * (seed + 1))
            for i in range(c1)
        ]

    def challenge_length(self) -> int:
        return self.n

    def response_length(self) -> int:
        return self.n

    def eval(self, challenges: ndarray) -> ndarray:
        N, n = challenges.shape
        assert n == self.n, f'Received challenges of length {n}, but expected {self.n}.'
        r, c, p = self.r, self.c, self.p
        s = ones(shape=(N, r + c))

        for i in range(p):
            s[:, :r] = s[:, :r] * challenges[:, r*i:r*(i+1)]
            s = self.apply_F(s)
        ret = ones(shape=(N, n))
        for i in range(p):
            ret[:, r*i:r*(i+1)] = s[:, :r]
            s = self.apply_F(s)
        return ret

    def apply_F(self, s):
        N, n = s.shape
        assert n == self.n
        c, c1, r = self.c, self.c1, self.r

        # Adjust capacity with PUFs
        for j in range(N):
            for i in range(c1):
                s[j, r + i] = self.pufs[i].eval(s[j, :c].reshape((1, c))).reshape((1,))

        # Permutation
        s[:, :] = s[:, self.f]

        return s


class SpongeReliabilityExperiment(Experiment):
    reliability: float

    def run(self):
        n, r, c, p = self.parameters.n, self.parameters.r, self.parameters.c, self.parameters.p
        sponge = Sponge(n, r, c, self.parameters.c1, p, self.parameters.transform, self.parameters.seed, self.parameters.noisiness)
        challenges = random_inputs(n, 100, RandomState(0))
        expectations = []
        for c in challenges:
            c_copied = broadcast_to(c, shape=(10, n))
            responses = sponge.eval(c_copied)
            expectations.append(average(absolute(average(responses, axis=0))))
        self.reliability = average(expectations)

    def analyze(self):
        return Result(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            reliability=self.reliability,
        )


class SpongeReliabilityStudy(Study):

    SHUFFLE = True

    def experiments(self):
        return [
            SpongeReliabilityExperiment(
                None,
                Parameters(
                    n=n, r=n-c, c=c, c1=c1, p=n//(n-c), transform=transform,
                    seed=seed, noisiness=noisiness,
                )
            )
            for n, c, c1_set in [
                (64, 32, [1, 4, 8, 16, 32]),  # p = 64 / 32 = 2
                (64, 48, [1, 8, 16, 32, 48]),  # p = 64 / 16 = 4
                (64, 56, [1, 8, 16, 32]),  # p = 64 / 8 = 8
                (128, 64, [1, 4, 8, 32]),  # p = 128 / 64 = 2
                (128, 96, [1, 4, 8, 32]),  # p = 128 / 32 = 4
                (128, 112, [1, 4, 8, 32]),  # p = 128 / 16 = 8
            ]
            for c1 in c1_set
            for seed in range(10)
            for transform in ['fixed_permutation']
            for noisiness in [.05, .1, .2]
        ]

    def plot(self):
        f = catplot(
            data=self.experimenter.results,
            x='c1',
            y='reliability',
            col='noisiness',
            row='p',
            hue='n',
        )
        f.savefig(f'figures/{self.name()}.pdf')
