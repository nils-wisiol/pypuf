from itertools import product
from typing import NamedTuple
from uuid import UUID

from matplotlib.pyplot import subplots
from numpy import ndarray, zeros, prod, sign
from numpy.random.mtrand import RandomState
from seaborn import catplot, heatmap

from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.pac.fourier_approximation import FourierCoefficientApproximation
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.simulation.base import Simulation
from pypuf.studies.base import Study
from pypuf.tools import TrainingSet, approx_dist, approx_dist_real, MonomialFactory


class Parameters(NamedTuple):
    n: int
    k_up: int
    k_down: int
    instance_seed: int


class Result(NamedTuple):
    experiment_id: UUID
    measured_time: float
    accuracy: float
    accuracy_up: float
    dist_up: float


class InterposePUFApproximationExperiment(Experiment):

    class XORSimulation(Simulation):

        def __init__(self, simulations) -> None:
            super().__init__()
            assert len(simulations), 'Must give at least one simulation to determine challenge_length.'
            self.n = simulations[0].challenge_length()
            for m in simulations:
                assert m.challenge_length() == self.n, 'All simulations must have the same challenge_length'
            self.simulations = simulations

        def eval(self, challenges: ndarray) -> ndarray:
            return sign(self.val(challenges))

        def val(self, challenges: ndarray) -> ndarray:
            return prod([m.val(challenges) for m in self.simulations], axis=0)

        def challenge_length(self) -> int:
            return self.n

        def response_length(self) -> int:
            return 1

    class InterposePUFApproximation(InterposePUF):
        def __init__(self, up_approximation, down_simulation, n: int, k_down: int, k_up: int = 1, interpose_pos: int = None, seed: int = None,
                     transform=None, noisiness=0, noise_seed=None):
            super().__init__(n, k_down, k_up, interpose_pos, seed, transform, noisiness, noise_seed)
            self.up = up_approximation
            self.down = down_simulation

        def eval(self, challenges, **kwargs):
            return super().eval(challenges)

        def _interpose_bits(self, challenges):
            (N, n) = challenges.shape
            return self.up.val(challenges).reshape(N, 1)

    def __init__(self, progress_log_name, parameters):
        super().__init__(progress_log_name, parameters)
        self.instance = None
        self.approximation = None
        self.up_chain_approximations = []

    def prepare(self):
        self.instance = InterposePUF(
            n=self.parameters.n,
            k_down=self.parameters.k_down,
            k_up=self.parameters.k_up,
            seed=self.parameters.instance_seed,
            transform=LTFArray.transform_atf  # ATF is important! Do not use transform_id.
        )

    def run(self):
        # We first estimate the Fourier coefficients of the upper XOR Arbiter PUF
        # We do so individually to make it more efficient
        for l in range(self.parameters.k_up):
            # prepare a single model for the l-th chain
            weight_array = self.instance.up.weight_array[l:l+1, :-1]
            chain = LTFArray(weight_array, LTFArray.transform_atf, LTFArray.combiner_xor)

            # approximate the "ATF" Fourier coefficients of this model
            monomials = MonomialFactory.monomials_atf(self.parameters.n)

            # the following block can be removed after rebase to version with BiPoly
            chi_set = []
            for m in monomials.keys():
                z = zeros(self.parameters.n)
                z[list(m)] = 1
                chi_set.append(z)

            self.up_chain_approximations.append(FourierCoefficientApproximation(
                training_set=TrainingSet(chain, 10000, RandomState(0)),
                chi_set=chi_set,
            ).learn())

        # The upper XOR Arbiter PUF will be approximated by the product of the individual chain simulations as estimated
        # above. The sign will not be taken!
        # To approximate the Interpose PUF, we will use the approximations of the upper chain and the down simulation
        # of self.instance
        self.approximation = self.InterposePUFApproximation(
            up_approximation=self.XORSimulation(self.up_chain_approximations),
            down_simulation=self.instance.down,
            n=self.parameters.n,
            k_down=self.parameters.k_down,
            k_up=self.parameters.k_up
        )

    def analyze(self):
        # estimate the accuracy of the upper approximation when taking the sign of the result
        approximation_up = self.XORSimulation(self.up_chain_approximations)
        instance_up = LTFArray(
            self.instance.up.weight_array[:, :self.parameters.n],
            transform=LTFArray.transform_atf,
            combiner=LTFArray.combiner_xor,
        )
        accuracy_up = 1 - approx_dist(instance_up, approximation_up, 2000, random_instance=RandomState(1))

        # estimate the distance to the instance value for the approximation of the upper XOR Arbiter PUF
        approximation_up.eval = approximation_up.val
        dist_up = approx_dist_real(instance_up, approximation_up, 2000, RandomState(2)) / 2

        # estimate total accuracy of the approximation
        accuracy = 1 - approx_dist(self.instance, self.approximation, 2000, random_instance=RandomState(3))

        # return result
        return Result(
            experiment_id=self.id,
            measured_time=self.measured_time,
            accuracy=accuracy,
            accuracy_up=accuracy_up,
            dist_up=dist_up,
        )


class InterposePUFApproximation(Study):

    COMPRESSION = True

    def experiments(self):
        return [
            InterposePUFApproximationExperiment(
                progress_log_name='',
                parameters=Parameters(
                    n=n,
                    k_up=k_up,
                    k_down=k_down,
                    instance_seed=i,
                )
            )
            for i in range(10)
            for n in [16, 32, 64, 128, 256]
            for k_up, k_down in list(product([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])) + [(1, 6), (1, 7), (1, 8), (1, 9)]
        ]

    def plot(self):
        data = self.experimenter.results

        f, _ = subplots(1, 1)
        heatmap(
            data=data.groupby(['n', 'k_up'], as_index=True).mean().reset_index().pivot(
                'n', 'k_up', 'accuracy_up'),
            annot=True,
            vmin=.5,
            vmax=1,
            ax=f.axes[0]
        )
        f.suptitle('Average Accuracy of n-bit k-XOR Arbiter PUF\nApproximation by Sign of Chow-Parameters')
        f.savefig(f'figures/{self.name()}.xor-arbiter-puf.accuracy.pdf')
        f.savefig(f'figures/{self.name()}.xor-arbiter-puf.accuracy.png')

        f, _ = subplots(1, 1)
        heatmap(
            data=data.groupby(['n', 'k_up'], as_index=True).mean().reset_index().pivot(
                'n', 'k_up', 'dist_up'),
            annot=True,
            vmin=0,
            vmax=1,
            ax=f.axes[0],
            cmap='rocket_r'
        )
        f.suptitle('Average Distance of n-bit k-XOR Arbiter PUF\nApproximation by Chow-Parameters')
        f.savefig(f'figures/{self.name()}.xor-arbiter-puf.dist.pdf')
        f.savefig(f'figures/{self.name()}.xor-arbiter-puf.dist.png')

        f, _ = subplots(1, 1)
        heatmap(
            data=data.groupby(['k_up', 'k_down'], as_index=True).mean().reset_index().pivot(
                'k_down', 'k_up', 'accuracy'),
            annot=True,
            vmin=.5,
            vmax=1,
            ax=f.axes[0],
            cmap='rocket',
        )
        f.suptitle('Average Accuracy of (k_up,k_down)-iPUF Approximation')
        f.savefig(f'figures/{self.name()}.heatmap.pdf')
        f.savefig(f'figures/{self.name()}.heatmap.png')

        g = catplot(
            data=data,
            x='n',
            y='accuracy',
            col='k_up',
            row='k_down',
            kind='box',
        )
        g.set(ylim=[.5, 1])
        f.suptitle('Accuracy Distribution of (k_up,k_down)-iPUF Approximation')
        g.savefig(f'figures/{self.name()}.detailed.pdf')
