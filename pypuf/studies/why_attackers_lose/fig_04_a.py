"""
Figure 4 (a) of "Why attackers lose: design and security analysis of arbitrarily large
XOR arbiter PUFs", Accepted 26 Feb 2019 Journal of Cryptographic Engineering.

This study examines the minimum number of votes needed such
that for a uniformly random challenge c we have Pr[Stab(c) ≥ 95%] ≥
80% for different k, as determined by a simulation (Sect. 6.2). The
simulation uses arbiter chain length of n = 32; however, we showed
that the results are independent of n. This log–log graph confirms the
result that the number of votes required grows polynomially.
"""
from matplotlib import pyplot
from matplotlib.ticker import FixedLocator, ScalarFormatter
from seaborn import lineplot, scatterplot

from pypuf.experiments.experiment.majority_vote import ExperimentMajorityVoteFindVotes, Parameters
from pypuf.studies.base import Study


class NumberOfVotesRequiredStudy(Study):
    """
    Generates Figure 4 (a) of "Why attackers lose: design and security analysis of arbitrarily large XOR arbiter PUFs"
    """

    SHUFFLE = True
    COMPRESSION = True

    RESTARTS = 200
    K_RANGE = 2
    K_MAX = 32
    LOWERCASE_N = 32
    UPPERCASE_N = 2000
    S_RATIO = .033
    ITERATIONS = 10
    SEED_CHALLENGES = 0xf000
    STAB_C = .95
    STAB_ALL = .80

    def experiments(self):
        e = []
        for i in range(self.RESTARTS):
            for k in range(self.K_RANGE, self.K_MAX + 1, self.K_RANGE):
                e.append(ExperimentMajorityVoteFindVotes(
                    progress_log_prefix=None,
                    parameters=Parameters(
                        n=self.LOWERCASE_N,
                        k=k,
                        challenge_count=self.UPPERCASE_N,
                        seed_instance=0xC0DEBA5E + i,
                        seed_instance_noise=0xdeadbeef + i,
                        transformation='id',
                        combiner='xor',
                        mu=0,
                        sigma=1,
                        sigma_noise_ratio=self.S_RATIO,
                        seed_challenges=self.SEED_CHALLENGES + i,
                        desired_stability=self.STAB_C,
                        overall_desired_stability=self.STAB_ALL,
                        minimum_vote_count=1,
                        iterations=self.ITERATIONS,
                        bias=None
                    )
                ))
        return e

    def plot(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xscale='log', yscale='log')
        ax.xaxis.set_major_locator(FixedLocator([2, 4, 6, 8, 12, 16, 20, 24, 28, 32]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
        ax.yaxis.set_major_formatter(ScalarFormatter())

        r = self.experimenter.results[['k', 'vote_count']].groupby(['k']).mean().reset_index()
        lineplot(
            x='k', y='vote_count', data=r,
            ax=ax, estimator=None, ci=None
        )
        scatterplot(
            x='k', y='vote_count', data=r, ax=ax
        )
        fig = ax.get_figure()
        fig.set_size_inches(6, 2.5)
        ax.set_xlabel('number of arbiter chains in the MV XOR Arbiter PUF')
        ax.set_ylabel('number of votes')
        ax.set_title('Number of votes required for Pr[Stab(c)>95%] > 80%')
        fig.savefig('figures/{}.pdf'.format(self.name()), bbox_inches='tight', pad_inches=0)
