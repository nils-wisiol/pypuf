"""
Studies systematic bias of XOR Arbiter PUFs and its variants.
"""
from itertools import combinations
from os import getpid
from typing import NamedTuple
from uuid import UUID

import sp80022suite
from matplotlib.pyplot import hist, subplots
from numpy import average, sign, arange, zeros, int8
from numpy.random.mtrand import RandomState
from seaborn import FacetGrid, heatmap

from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.studies.base import Study
from pypuf.tools import random_inputs


class Parameters(NamedTuple):
    """
    Defines the type of XOR Arbiter PUF to be tested for bias
    """
    n: int
    k: int
    transform: str
    N: int
    seed: int


class Result(NamedTuple):
    """
    Defines the result of the bias test.
    """
    experiment_id: UUID
    pid: int
    measured_time: float
    bias: float
    p_bias: float
    uniqueness: str
    p_value: float
    test_passed: bool


class BiasExperiment(Experiment):
    """
    Tests a defined XOR Arbiter PUF (variant) instance for systematic bias.
    """

    instance = None
    individual_instances = None
    responses = None
    uniqueness = None

    def prepare(self):
        self.instance = XORArbiterPUF(
            self.parameters.n,
            self.parameters.k,
            self.parameters.seed,
            self.parameters.transform,
        )
        self.individual_instances = [
            LTFArray(
                weight_array=weights[:self.parameters.n].reshape(1, self.parameters.n),
                transform=self.parameters.transform,
                combiner='xor',
            )
            for weights in self.instance.weight_array
        ]

    def run(self):
        inputs = random_inputs(self.parameters.n, self.parameters.N, RandomState(self.parameters.seed))
        self.responses = self.instance.val(inputs)
        self.uniqueness = zeros(shape=(self.parameters.k, self.parameters.k))
        for (idx1, i1), (idx2, i2) in combinations(enumerate(self.individual_instances), 2):
            self.uniqueness[idx1, idx2] = average(i1.eval(inputs) * i2.eval(inputs))

    def analyze(self):
        p = sp80022suite.frequency(bytes(((1 - sign(self.responses)) / 2).astype(int8)))
        return Result(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            p_bias=average(self.responses),
            bias=average(sign(self.responses)),
            uniqueness=str(self.uniqueness),
            p_value=p,
            test_passed=p >= .01,
        )


class BiasDistributionStudy(Study):
    """
    Studies XOR Arbiter PUFs and variants of different sizes for systematic bias,
    based on a large sample size for each type.
    """

    SHUFFLE = True
    COMPRESSION = True

    def experiments(self):
        return [
            BiasExperiment(
                progress_log_name='',
                parameters=Parameters(
                    n=n, k=k, transform=transform, N=10**6, seed=seed
                )
            )
            for n in [64]
            for k in [1, 2, 3, 4]
            for transform in ['atf', 'lightweight_secure', 'fixed_permutation']
            for seed in range(5000)
        ]

    def plot(self):
        # prepare data
        data = self.experimenter.results.copy()
        data['test_passed'] = data.apply(lambda row: int(row['test_passed']), axis=1)
        data['transform'] = data['transform'].replace('atf', '(None)')
        data['transform'] = data['transform'].replace('lightweight_secure', 'LW Sec')
        data['transform'] = data['transform'].replace('fixed_permutation', 'Perm.')

        # size parameters
        ncol = 5
        nrow = 4
        height = 1.2
        aspect = 1.4

        # plot bias histograms
        bin_size = .005
        bin_range = .08
        grid = FacetGrid(data, col='k', row='transform', height=height, aspect=aspect, sharey=False,
                         gridspec_kws={'hspace': .5, 'wspace': .05}, row_order=['(None)', 'LW Sec', 'Perm.'])
        grid.map(
            hist,
            'bias',
            density=True,
            bins=arange(-bin_range - .5 * bin_size, bin_range, bin_size)
        )
        grid.set_yticklabels([])
        for ax in grid.axes.flat:
            ax.set_yticks([], [])
        grid.set_titles(template='{row_name}, k={col_name}')
        grid.savefig(f'figures/{self.name()}.pdf')
        grid.savefig(f'figures/{self.name()}.png')

        # plot portion of instances that pass NIST test
        f, _ = subplots(1, 1, figsize=(ncol * height * aspect / 1.5, nrow * height / 1.5))
        heatmap(
            data=data.groupby(['transform', 'k'], as_index=True).mean().reset_index().pivot(
                'transform', 'k', 'test_passed'),
            annot=True,
            vmin=0,
            vmax=1,
            ax=f.axes[0],
            cmap='rocket',
        )
        f.tight_layout()
        f.savefig(f'figures/{self.name()}.test_scores.pdf')
        f.savefig(f'figures/{self.name()}.test_scores.png')
