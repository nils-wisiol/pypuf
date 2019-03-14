"""
Figure 4 (b) of "Why attackers lose: design and security analysis of arbitrarily large
XOR arbiter PUFs", Accepted 26 Feb 2019 Journal of Cryptographic Engineering.

This study generates a histogram showing the probability density of an XOR Majority Vote
Arbiter PUF of size k = 32 and chain length of n = 32. We used 51 and 501 votes to boost
stability to Pr[Stab(c) ≥ 95%] ≥ 80% and to the stability of the building block arbiter
chains, respectively. The dashed line shows the theoretical stability probability density
for a single arbiter chain (i.e., before majority vote and XOR) as used in this simulation
( σ Noise / σ Model = 0.033). The graph confirms that a Majority Vote XOR Arbiter PUF built
from these arbiter chains and the given number of votes cannot only achieve a decent stability
(at 51 votes), but also reach the same stability as a single arbiter chain (at 501 votes).
"""
from os import getpid
from typing import NamedTuple, List
from uuid import UUID

from matplotlib.pyplot import figure
from numpy import arange, exp
from numpy.random.mtrand import RandomState
from seaborn import distplot, lineplot
from scipy.special import erfinv

from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.ltfarray import NoisyLTFArray, LTFArray, SimulationMajorityLTFArray
from pypuf.studies.base import Study
from pypuf.tools import approx_stabilities


class Parameters(NamedTuple):
    """
    Parameters for StabilityExperiment.
    """
    n: int
    k: int
    sigma_noise_ratio: float
    seed: int
    vote_count: int
    N: int
    samples: int


class Result(NamedTuple):
    """
    Result of StabilityExperiment.
    """
    experiment_id: UUID
    pid: int
    stability: List[float]


class StabilityExperiment(Experiment):
    """
    Examines the stability of an LTFArray as defined in the given parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stability = None

    def run(self):
        random = RandomState(seed=self.parameters.seed)
        sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(
            self.parameters.n, 1, self.parameters.sigma_noise_ratio)
        weights = LTFArray.normal_weights(self.parameters.n, self.parameters.k, random_instance=random)
        instance_mv = SimulationMajorityLTFArray(weights,
                                                 LTFArray.transform_atf,
                                                 LTFArray.combiner_xor,
                                                 sigma_noise,
                                                 random_instance_noise=random,
                                                 vote_count=self.parameters.vote_count)

        self.stability = approx_stabilities(instance_mv, self.parameters.N, self.parameters.samples, random)

    def analyze(self):
        return Result(
            experiment_id=self.id,
            pid=getpid(),
            stability=self.stability.tolist(),
        )


class StabilityStudy(Study):
    """
    Generates Figure 4 (a) of "Why attackers lose: design and security analysis of arbitrarily large XOR arbiter PUFs"
    """

    SHUFFLE = True
    COMPRESSION = True

    COLORS = ['darkred', 'gold', 'navy']

    VOTES = [51, 501]
    SIZE = (32, 32)
    SIGMA_NOISE_RATIO = 0.033
    N = 10000
    SAMPLES = 200
    SEED = 0xbeef

    def experiments(self):
        (n, k) = self.SIZE
        e = []
        for votes in self.VOTES:
            e.append(
                StabilityExperiment(
                    progress_log_name=None,
                    parameters=Parameters(
                        n=n, k=k,
                        sigma_noise_ratio=self.SIGMA_NOISE_RATIO,
                        seed=self.SEED,
                        vote_count=votes,
                        N=self.N,
                        samples=self.SAMPLES,
                    )
                )
            )
        return e

    def plot(self):
        fig = figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim([.2 * 1e-3, 10])
        ax.set(yscale='log')

        rows = self.experimenter.results[['vote_count', 'stability']].sort_values(['vote_count']).iterrows()
        for index, row in rows:
            stab = row['stability']
            if isinstance(stab, str):
                stab = [float(x.strip(' []')) for x in stab.split(',')]
            elif not isinstance(stab, list) or not isinstance(stab[0], float):
                raise Exception('Cannot read stability list for vote count {}, it has type {}[{}]'.format(
                    row['vote_count'],
                    type(stab),
                    type(stab[0])
                ))
            ax = distplot(
                stab,
                norm_hist=True,
                kde=False,
                label='{} votes'.format(row['vote_count']),
                hist_kws={'color': self.COLORS[index], 'alpha': 1},
            )

        sample_width = .0001
        plot_range = arange(.5 + sample_width, 1.00, sample_width)
        offset = .01 / 2
        lineplot(
            x=plot_range,
            y=[
                # This is the PDF of the distribution defined by CDF(z) = erf(self.SIGMA_NOISE_RATIO * erfinv(2z - 1))
                # To match the buckets of the histogram, we offset the graph by half a bucket
                2 * self.SIGMA_NOISE_RATIO * exp(erfinv(2 * (z + offset) - 1) ** 2 * (1 - self.SIGMA_NOISE_RATIO ** 2))
                for z in plot_range
            ],
            ax=ax,
            label='Building block stability σNoise/σModel = {}'.format(self.SIGMA_NOISE_RATIO),
            color=self.COLORS[-1],
        )

        fig = ax.get_figure()
        fig.set_size_inches(6, 2.5)
        ax.legend(loc=4)
        ax.set_xlabel('stability')
        ax.set_ylabel('rel. frequency')
        ax.set_title('Stability frequencies / stability probability density')
        fig.savefig('figures/{}.pdf'.format(self.name()), bbox_inches='tight', pad_inches=0)
