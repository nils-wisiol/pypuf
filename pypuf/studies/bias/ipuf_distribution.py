"""
Studies systematic bias of Interpose PUFs of various sizes.
"""
from os import getpid
from typing import NamedTuple
from uuid import UUID

import sp80022suite
from matplotlib.pyplot import hist, subplots
from numpy import average, arange, int8
from numpy.random.mtrand import RandomState
from seaborn import FacetGrid, heatmap

from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF
from pypuf.studies.base import Study
from pypuf.tools import random_inputs


class Parameters(NamedTuple):
    """
    Defines the Interpose PUF to be tested.
    """
    n: int
    k_up: int
    k_down: int
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
    p_value: float
    test_passed: bool


class BiasExperiment(Experiment):
    """
    Tests a defined Interpose PUF instance for systematic bias.
    """

    instance = None
    responses = None

    def prepare(self):
        self.instance = InterposePUF(
            self.parameters.n,
            self.parameters.k_down,
            self.parameters.k_up,
            seed=self.parameters.seed,
        )

    def run(self):
        self.responses = self.instance.eval(random_inputs(
            self.parameters.n, self.parameters.N, RandomState(self.parameters.seed)))

    def analyze(self):
        p = sp80022suite.frequency(bytes(((1 - self.responses) / 2).astype(int8)))
        return Result(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            bias=average(self.responses),
            p_value=p,
            test_passed=p >= .01,
        )


class BiasDistributionStudy(Study):
    """
    Empirically tests Interpose PUFs of various sizes for systematic bias based
    on a large sample size for each size.
    """

    SHUFFLE = True
    COMPRESSION = True

    def experiments(self):
        return [
            BiasExperiment(
                progress_log_name='',
                parameters=Parameters(n, k_up, k_down, 10**6, seed)
            )
            for n in [64]
            for k_up in range(1, 6)
            for k_down in range(1, 5)
            for seed in range(5000)
        ]

    def plot(self):
        # prepare data
        data = self.experimenter.results.copy()
        data['test_passed'] = data.apply(lambda row: int(row['test_passed']), axis=1)

        # size parameters
        ncol = 5
        nrow = 4
        height = 1.2
        aspect = 1.4

        # plot histograms
        bin_size = .005
        bin_range = .08
        grid = FacetGrid(data, col='k_up', row='k_down', height=height, aspect=aspect, sharey=False,
                         gridspec_kws={'hspace': .5, 'wspace': .05})
        grid.map(
            hist,
            'bias',
            density=True,
            bins=arange(-bin_range - .5 * bin_size, bin_range, bin_size)
        )
        grid.set_yticklabels([])
        for ax in grid.axes.flat:
            ax.set_yticks([], [])
        grid.set_titles(template='({row_name},{col_name})')
        grid.savefig(f'figures/{self.name()}.pdf')
        grid.savefig(f'figures/{self.name()}.png')

        # plot portion of instances that pass the NIST test
        f, _ = subplots(1, 1, figsize=(ncol * height * aspect / 1.5, nrow * height / 1.5))
        heatmap(
            data=data.groupby(['k_up', 'k_down'], as_index=True).mean().reset_index().pivot(
                'k_down', 'k_up', 'test_passed'),
            annot=True,
            vmin=0,
            vmax=1,
            ax=f.axes[0],
            cmap='rocket',
        )
        f.tight_layout()
        f.savefig(f'figures/{self.name()}.test_scores.pdf')
        f.savefig(f'figures/{self.name()}.test_scores.png')
