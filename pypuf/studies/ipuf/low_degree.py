from os import getpid
from typing import NamedTuple
from uuid import UUID

from numpy.random.mtrand import RandomState
from seaborn import catplot

from pypuf.bipoly import BiPoly
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.pac.fourier_approximation import FourierCoefficientApproximation
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF, LightweightSecurePUF
from pypuf.studies.base import Study
from pypuf.tools import TrainingSet, approx_dist


class Parameter(NamedTuple):
    n: int
    k: int
    seed: int
    N: int
    transform: str


class Result(NamedTuple):
    experiment_id: UUID
    pid: int
    measured_time: float
    length: int
    accuracy: float


class PTFFirstDegreeAttackExperiment(Experiment):

    instance = None
    polynomial = None
    training_set = None
    model = None

    def instance_id(self):
        return (
            XORArbiterPUF(n=self.parameters.n, k=self.parameters.k, seed=self.parameters.seed, transform='id'),
            (BiPoly.linear(n=self.parameters.n)**self.parameters.k).to_vector_notation(n=self.parameters.n),
        )

    def instance_atf(self):
        return (
            XORArbiterPUF(n=self.parameters.n, k=self.parameters.k, seed=self.parameters.seed, transform='atf'),
            BiPoly.xor_arbiter_puf(n=self.parameters.n, k=self.parameters.k).to_vector_notation(n=self.parameters.n),
        )

    def instance_lightweight_secure(self):
        return (
            LightweightSecurePUF(n=self.parameters.n, k=self.parameters.k, seed=self.parameters.seed),
            BiPoly.lightweight_secure_puf(n=self.parameters.n, k=self.parameters.k).to_vector_notation(
                n=self.parameters.n),
        )

    def instance_fixed_permutation(self):
        return (
            XORArbiterPUF(n=self.parameters.n, k=self.parameters.k, seed=self.parameters.seed,
                          transform='fixed_permutation'),
            BiPoly.permutation_puf(n=self.parameters.n, k=self.parameters.k).to_vector_notation(
                n=self.parameters.n),
        )

    def prepare(self):
        self.instance, self.polynomial = getattr(self, f'instance_{self.parameters.transform}')()
        self.training_set = TrainingSet(self.instance, self.parameters.N, RandomState(seed=self.parameters.seed))

    def run(self):
        self.model = FourierCoefficientApproximation(self.training_set, self.polynomial).learn()

    def analyze(self):
        return Result(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            length=len(self.polynomial),
            accuracy=1 - approx_dist(self.instance, self.model, 10**4, RandomState(seed=0)),
        )


class PTFFirstDegreeAttackStudy(Study):

    SHUFFLE = True

    def experiments(self):
        return [
            PTFFirstDegreeAttackExperiment(
                progress_log_name='',
                parameters=Parameter(
                    n=n, k=k, seed=seed, N=N, transform=transform
                )
            )
            for n in [64, 128, 256, 512]
            for k in [1]
            for N in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
            for transform in ['atf']
            for seed in range(100)
        ] + [
            PTFFirstDegreeAttackExperiment(
                progress_log_name='',
                parameters=Parameter(
                    n=n, k=k, seed=seed, N=N, transform=transform
                )
            )
            for n in [64]
            for k in [1, 2, 3]
            for N in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
            for transform in ['atf', 'lightweight_secure', 'fixed_permutation']
            for seed in range(10)
        ]

    def plot(self):
        for y in ['accuracy', 'measured_time']:
            f = catplot(
                data=self.experimenter.results,
                x='N',
                y=y,
                hue='n',
                col='k',
                row='transform',
                kind='boxen',
                aspect=2,
                height=3,
            )
            f.savefig(f'figures/{self.name()}.{y}.pdf')
