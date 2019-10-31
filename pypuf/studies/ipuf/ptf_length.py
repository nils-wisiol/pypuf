from typing import NamedTuple
from uuid import UUID

from matplotlib.ticker import ScalarFormatter
from pandas import melt
from seaborn import relplot, axes_style

from pypuf.bipoly import BiPoly
from pypuf.experiments.experiment.base import Experiment
from pypuf.studies.base import Study


class Parameters(NamedTuple):
    n: int
    k: int


class Result(NamedTuple):
    experiment_id: UUID
    measured_time: float
    length_xor_arbiter_puf: int
    length_lightweight_secure_puf: int
    length_permutation_puf: int
    length_ipuf_approx: int
    max_memory: int


class PTFLengthExperiment(Experiment):

    def __init__(self, progress_log_name, parameters):
        super().__init__(progress_log_name, parameters)
        self.length_xor_arbiter_puf = None
        self.length_lightweight_secure_puf = None
        self.length_permutation_puf = None
        self.length_ipuf_approx = None

    def run(self):
        n, k = self.parameters.n, self.parameters.k
        xor_arbiter_puf = BiPoly.xor_arbiter_puf(n, k)
        self.length_xor_arbiter_puf = len(xor_arbiter_puf)
        self.length_lightweight_secure_puf = len(BiPoly.lightweight_secure_puf(n, k))
        self.length_permutation_puf = len(BiPoly.permutation_puf(n, k))
        self.length_ipuf_approx = len(BiPoly.interpose_puf_approximation(n, k_up=k, k_down=1, p_up=xor_arbiter_puf))

    def analyze(self):
        return Result(
            experiment_id=self.id,
            measured_time=self.measured_time,
            length_xor_arbiter_puf=self.length_xor_arbiter_puf,
            length_lightweight_secure_puf=self.length_lightweight_secure_puf,
            length_permutation_puf=self.length_permutation_puf,
            length_ipuf_approx=self.length_ipuf_approx,
            max_memory=self.max_memory(),
        )


class InterposePUFApproxPTFLengthStudy(Study):

    COMPRESSION = True

    def experiments(self):
        return [
            PTFLengthExperiment(
                progress_log_name='',
                parameters=Parameters(
                    n=n,
                    k=k,
                )
            )
            for k, n_range in [
                (1, range(16, 256, 2)),
                (2, range(16, 256, 2)),
                (3, range(16, 128, 2)),
                (4, range(16, 72, 2)),
                (5, range(16, 64, 2)),
                (6, range(16, 32, 2)),
            ]
            for n in n_range
        ]

    def plot(self):
        # select data and unpivot
        lengths = [
            'length_xor_arbiter_puf',
            'length_lightweight_secure_puf',
            'length_permutation_puf',
            'length_ipuf_approx',
        ]
        data = melt(
            self.experimenter.results[['n', 'k'] + lengths],
            id_vars=['n', 'k'],
            value_vars=lengths,
            var_name='PUF',
            value_name='PTF Length',
        )

        # human-readable type
        data['PUF'] = data.apply(
            lambda row: {
                'length_xor_arbiter_puf': 'XOR Arb PUF',
                'length_lightweight_secure_puf': 'LW Sec PUF',
                'length_permutation_puf': 'Perm. PUF',
                'length_ipuf_approx': 'iPUF Approx.',
            }[row['PUF']],
            axis=1,
        )

        with axes_style('whitegrid'):
            g = relplot(
                x='n',
                y='PTF Length',
                style='k',
                hue='PUF',
                data=data,
                legend='full',
                kind='line',
                markers=False,
                ci=None,  # no error bars
            )
            for ax in g.axes.flatten():
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xticks([4, 8, 16, 24, 32, 48, 64, 128, 256])
                ax.get_xaxis().set_major_formatter(ScalarFormatter())
                ax.set_xlim([16, 256 + 10])
                ax.set_ylim([8, .5 * 10 ** 8])
            g.fig.set_size_inches(7, 2)
            g.savefig('figures/%s.pdf' % self.name())

            g.fig.suptitle('PTF Lengths for n-bit (1,k)-Interpose PUF Approximation\n'
                           'and n-bit k-XOR Arbiter PUFs')
            g.fig.subplots_adjust(top=0.85)
            g.fig.set_size_inches(7, 3)
            g.savefig('figures/%s.png' % self.name())
