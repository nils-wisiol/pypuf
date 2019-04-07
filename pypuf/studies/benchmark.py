"""
Benchmark the LTFArray.eval method.
"""
import abc
import sys

from matplotlib.pyplot import subplots

from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, CompoundTransformation
from pypuf.simulation.weak import NoisySRAM
from pypuf.studies.base import Study
from pypuf.tools import sample_inputs, TrainingSet
from numpy.distutils import cpuinfo
from numpy.random.mtrand import RandomState
from os import getpid
from typing import NamedTuple, Union
from uuid import UUID
from seaborn import barplot


class LTFBenchmarkParameters(NamedTuple):
    """
    Experiment parameters for LTFArray.eval benchmark.
    """
    version: Union[str, None]
    cpu: Union[str, None]
    benchmark_group: str
    n: int
    k: int
    N: int
    transform: str
    combiner: str
    seed_input: int


class BenchmarkResult(NamedTuple):
    """
    Experiment result from LTFArray.eval benchmark.
    """
    experiment_id: UUID
    pid: int
    measured_time: float


class BenchmarkExperiment(Experiment):
    """
    A benchmark experiment; the result is just the basic information
    and the measured time.
    """

    def analyze(self):
        return BenchmarkResult(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
        )

    @abc.abstractmethod
    def run(self):
        pass


class LTFBenchmarkExperiment(BenchmarkExperiment):
    """
    Measures the time LTFArray.eval takes to evaluate a set of inputs.
    """

    def __init__(self, progress_log_prefix, parameters):
        super().__init__(progress_log_prefix, parameters)
        self.set = None
        self.ltf_array = None

    def prepare(self):
        self.set = sample_inputs(self.parameters.n, self.parameters.N, RandomState(seed=self.parameters.seed_input))
        self.ltf_array = LTFArray(
            weight_array=LTFArray.normal_weights(self.parameters.n, self.parameters.k),
            transform=self.parameters.transform,
            combiner=self.parameters.combiner,
        )

    def run(self):
        self.ltf_array.eval(self.set)


class LRBenchmarkExperiment(LTFBenchmarkExperiment):
    """
    Measures the time the logistic regression learner takes to learn a target.
    """

    def __init__(self, progress_log_prefix, parameters):
        super().__init__(progress_log_prefix, parameters)
        self.learner = None

    def prepare(self):
        super().prepare()
        self.learner = LogisticRegression(
            t_set=TrainingSet(instance=self.ltf_array, N=self.parameters.N),
            n=self.parameters.n,
            k=self.parameters.k,
            transformation=self.ltf_array.transform,
            combiner=self.ltf_array.combiner,
            convergence_decimals=100,  # never converge
            iteration_limit=20,
        )

    def run(self):
        self.learner.learn()


class Benchmark(Study):
    """
    Measure LTFArray.eval cpu time for a collection of input transformations and input set sizes.
    """

    TRANSFORMS = [
        'id',
        'atf',
        'lightweight_secure',
        'fixed_permutation',
        'random',
        CompoundTransformation(
            generator=LTFArray.generate_ipmod2_transform,
            args=(64, 8, NoisySRAM((8, 64, 64), .2, 31415)),
            name='transform_ipmod2_NoisySRAM_31415_.2',
        )
    ]
    SAMPLE_SIZE = 100
    SHUFFLE = True
    COMPRESSION = True

    def experiments(self):
        cpu = None
        try:
            cpu = cpuinfo.cpu.info[0]['model name']
        except Exception:  # pylint: disable=W
            pass

        experiments = []

        # benchmark LTFArray
        experiments.extend([
            LTFBenchmarkExperiment(
                progress_log_prefix=None,
                parameters=LTFBenchmarkParameters(
                    n=64,
                    k=8,
                    N=100000,
                    transform=transform,
                    combiner='xor',
                    seed_input=314159 + i,
                    version=sys.version,
                    cpu=cpu,
                    benchmark_group='LTF Array {}'.format(transform)
                )
            )
            for transform in self.TRANSFORMS
            for i in range(self.SAMPLE_SIZE)
        ])

        # benchmark LogisticRegression
        experiments.extend([
            LRBenchmarkExperiment(
                progress_log_prefix=None,
                parameters=LTFBenchmarkParameters(
                    n=64,
                    k=8,
                    N=10000,
                    transform=transform,
                    combiner='xor',
                    seed_input=314159 + i,
                    version=sys.version,
                    cpu=cpu,
                    benchmark_group='LR Attack {}'.format(transform)
                )
            )
            for transform in self.TRANSFORMS
            for i in range(self.SAMPLE_SIZE)
        ])

        return experiments

    def plot(self):
        # prepare data
        data = self.experimenter.results.sort_values('benchmark_group').copy()
        data['version'] = data.apply(lambda row: '.'.join(row['version'].split(' ')[0].split('.')[:2]), axis=1)
        data['cpu'] = data.apply(lambda row: row['cpu'].replace('Intel(R) Core(TM) ', ''), axis=1)
        data['cpu'] = data.apply(lambda row: row['cpu'].replace('Intel(R) Xeon(R) CPU ', ''), axis=1)
        data['platform'] = data.apply(lambda row: '{} using Python {}'.format(row['cpu'], row['version']), axis=1)
        data = data.sort_values(['benchmark_group', 'platform'])

        # plot
        fig, ax = subplots(1, 1)
        fig.set_size_inches(7, 7)

        barplot(
            y='benchmark_group',
            x='measured_time',
            hue='platform',
            data=data,
            ax=ax,
        )
        ax.set_xlabel('Mean Run Time')
        ax.set_ylabel('Benchmark')
        fig.suptitle('pypuf Benchmark Results')
        fig.savefig('figures/%s.pdf' % self.name(), bbox_inches='tight', pad_inches=.5)
