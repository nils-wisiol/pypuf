from os import getpid
from typing import NamedTuple
from uuid import UUID

from numpy import nonzero
from numpy.random import RandomState

from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.pac.fourier_approximation import FourierCoefficientApproximation
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF
from pypuf.studies.base import Study
from pypuf.tools import TrainingSet, GoldreichLevin, approx_dist


class Parameters(NamedTuple):
    n: int
    k: int
    transform: str
    N: int
    tau: float
    delta: float
    sample_size_override: int


class Result(NamedTuple):
    experiment_id: UUID
    pid: int
    measured_time: float
    n_coefficients: int
    coefficients: object
    accuracy: float
    weight_sample_size: int


class UnknownTransformAttackExperiment(Experiment):

    instance = None
    training_set = None
    coefficients = None
    model = None
    weight_sample_size = None

    def prepare(self):
        self.instance = XORArbiterPUF(
            n=self.parameters.n,
            k=self.parameters.k,
            seed=1,
            transform=self.parameters.transform,
        )
        self.training_set = TrainingSet(self.instance, self.parameters.N, RandomState(2))
        self.progress_logger.debug('Training set generated')

    def run(self):
        gl = GoldreichLevin(
            self.instance,
            self.parameters.tau,
            self.parameters.delta,
            self.parameters.sample_size_override,
        )
        self.weight_sample_size = gl.sample_size
        self.progress_logger.debug(f'Using {gl.sample_size} challenges for weight sampling')
        self.coefficients = gl.find_heavy_monomials(self.progress_logger)
        self.progress_logger.debug(f'found {len(self.coefficients)} coefficients using GoldreichLevin')
        if self.coefficients:
            self.model = FourierCoefficientApproximation(self.training_set, self.coefficients).learn()

    def analyze(self):
        if self.model:
            accuracy = 1 - approx_dist(self.instance, self.model, 10000, RandomState(3))
        else:
            accuracy = None

        return Result(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            n_coefficients=len(self.coefficients),
            coefficients=[nonzero(c) for c in self.coefficients],
            accuracy=accuracy,
            weight_sample_size=self.weight_sample_size,
        )


class UnknownTransformStudy(Study):

    SHUFFLE = True

    def experiments(self):
        return [
            UnknownTransformAttackExperiment(
                progress_log_name=f'unknown_transform_n={n}_k={k}_transform={transform}_N={N}_tau={tau}_delta={delta}',
                parameters=Parameters(
                    n=n,
                    k=k,
                    transform=transform,
                    N=N,
                    tau=tau,
                    delta=delta,
                    sample_size_override=10**5,
                )
            )
            for n in [64]
            for k in [1, 2]
            for transform in ['id', 'atf', 'lightweight_secure', 'fixed_permutation']
            for N in [10000]
            for tau in [.2, .3, .4]
            for delta in [.9, .8, .5, .3, .1]
        ]
