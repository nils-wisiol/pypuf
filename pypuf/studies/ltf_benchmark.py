"""
Benchmark the LTFArray.eval method.
"""
import sys
from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.studies.base import Study
from pypuf.tools import sample_inputs
from numpy.distutils import cpuinfo
from numpy.random.mtrand import RandomState
from os import getpid
from typing import NamedTuple, Union
from uuid import UUID
from seaborn import relplot


class Parameters(NamedTuple):
    """
    Experiment parameters for LTFArray.eval benchmark.
    """
    n: int
    k: int
    N: int
    transform: str
    combiner: str
    seed_input: int
    version: Union[str, None]
    cpu: Union[str, None]


class Result(NamedTuple):
    """
    Experiment result from LTFArray.eval benchmark.
    """
    experiment_id: UUID
    pid: int
    measured_time: float


class Benchmark(Experiment):
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

    def analyze(self):

        return Result(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
        )


class LTFBenchmark(Study):
    """
    Measure LTFArray.eval cpu time for a collection of input transformations and input set sizes.
    """

    Ns = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    TRANSFORMS = ['id', 'atf']
    SIZE = (64, 4)
    SAMPLE_SIZE = 100
    SHUFFLE = True

    def name(self):
        return 'ltf_benchmark'

    def experiments(self):
        (n, k) = self.SIZE
        cpu = None
        try:
            cpu = cpuinfo.cpu.info[0]['model name']
        except Exception:  # pylint: disable=W
            pass

        return [
            Benchmark(
                progress_log_prefix=None,
                parameters=Parameters(
                    n=n,
                    k=k,
                    N=N,
                    transform=transform,
                    combiner='xor',
                    seed_input=314159 + i,
                    version=sys.version,
                    cpu=cpu,
                )
            )
            for N in self.Ns
            for transform in self.TRANSFORMS
            for i in range(self.SAMPLE_SIZE)
        ]

    def plot(self):
        results = self.experimenter.results[['cpu', 'version', 'transform', 'N', 'measured_time']]
        groups = results.groupby(['transform'])

        for transform, group in groups:
            figure = relplot(x='N', y='measured_time', hue='cpu', style='version', kind='line', data=group)
            figure.savefig('figures/%s-%s.pdf' % (self.name(), transform))
