from typing import NamedTuple
from uuid import UUID

from matplotlib.pyplot import subplots
from numpy.random.mtrand import RandomState
from seaborn import heatmap, cm

from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.studies.base import Study
from pypuf.tools import TrainingSet, ChallengeResponseSet, approx_dist, MonomialFactory, \
    LinearizationModel, ModelFromLinearization


class Parameters(NamedTuple):
    N: int
    n: int
    k_up: int
    k_down: int
    instance_seed: int


class Result(NamedTuple):
    experiment_id: UUID
    measured_time: float
    accuracy: float
    monomial_count: int
    model: object
    memory_rss_max: int
    up_dist: float = -1
    up_dist_sign: float = -1


class InterposePUFLRAttackExperiment(Experiment):

    TEST_SET_MAX_SIZE = 5000
    TEST_SET_REL_SIZE = .1

    def __init__(self, progress_log_name, parameters: Parameters):
        super().__init__(progress_log_name, parameters)
        self.monomials = None
        self.training_set = None
        self.learner = None
        self.instance = None
        self.model = None

    def prepare(self):
        self.progress_logger.debug('computing monomials for n=%i k_up=%i' % (self.parameters.n, self.parameters.k_up))
        self.monomials = MonomialFactory.monomials_ipuf(self.parameters.n, self.parameters.k_up, 1)
        self.progress_logger.debug('computed %i monomials' % len(self.monomials))
        self.linearizer = LinearizationModel(self.monomials)
        self.progress_logger.debug('Instanciating Interpose PUF simulation')
        self.instance = InterposePUF(
            n=self.parameters.n,
            k_down=self.parameters.k_down,
            k_up=self.parameters.k_up,
            seed=self.parameters.instance_seed,
            transform=LTFArray.transform_atf  # ATF is important! Do not use transform_id.
        )
        self.progress_logger.debug('Creating challenge-response set with %i examples ...' % self.parameters.N)
        self.training_set = TrainingSet(self.instance, self.parameters.N, random_instance=RandomState(31415))
        self.progress_logger.debug(
            f'Created challenge-response set of size '
            f'{(self.training_set.challenges.nbytes + self.training_set.responses.nbytes) // 1024**3:.4f} GiB.'
        )

    def run(self):
        self.progress_logger.debug('Linearizing challenge-response set ...')
        linearized_example_set = ChallengeResponseSet(
            challenges=self.linearizer.linearize(self.training_set.challenges, logger=self.progress_logger),
            responses=self.training_set.responses,
        )
        self.progress_logger.debug('Splitting into training set and test set.')
        training_set_size = min(
            int(len(linearized_example_set.challenges) * self.TEST_SET_REL_SIZE),
            self.TEST_SET_MAX_SIZE
        )
        test_set = linearized_example_set.subset(slice(0, training_set_size))
        training_set = linearized_example_set.subset(slice(training_set_size, len(linearized_example_set.challenges)))
        self.progress_logger.debug('Starting learner ...')
        self.learner = LogisticRegression(
            t_set=training_set,
            test_set=test_set,
            n=len(self.monomials),
            k=self.parameters.k_down,
            transformation=LTFArray.transform_id,
            logger=self.progress_logger,
            iteration_limit=100 * self.parameters.k_down,
            weights_prng=RandomState(0),
        )
        inner_model = self.learner.learn()
        self.progress_logger.debug('Creating model from linearization')
        self.model = ModelFromLinearization(self.monomials, inner_model, self.parameters.n)


    def analyze(self):
        self.progress_logger.debug('Analyzing result')
        accuracy = 1 - approx_dist(self.instance, self.model, 10 ** 4, random_instance=RandomState(1))
        self.progress_logger.debug(f'final accuracy on instance: {accuracy}, '
                                   f'test set accuracy {1 - self.learner.test_set_dist}')
        self.progress_logger.debug(f'Peak memory usage was {self.max_memory()/1024**3:.4f} GiB')
        return Result(
            experiment_id=self.id,
            measured_time=self.measured_time,
            accuracy=accuracy,
            monomial_count=len(self.monomials),
            model=self.model.inner_model.weight_array,
            memory_rss_max=self.max_memory(),
        )


class InterposePUFLRAttack(Study):

    def experiments(self):
        return [
            attack(
                progress_log_name='%s_%i_%i_%i_%i_%i' % (attack.__name__, n, k_up, k_down, N, i),
                parameters=Parameters(
                    n=n,
                    k_up=k_up,
                    k_down=k_down,
                    N=N,
                    instance_seed=i,
                )
            )
            for i in range(2)
            for (n, k_up, k_down, Ns) in [
                # k_up = 1, effectively approximated by 1585-bit k=k_down XOR Arbiter PUF
                (64, 1, 1, [5 * 10**4]),  # approx mem .03 GiB
                (64, 1, 2, [5 * 10**4, 10 * 10**4]),  # approx mem .15 GiB
                (64, 1, 3, [10**6, 2 * 10**6]),  # approx mem 2.95 GiB
                (64, 1, 4, [20 * 10**6]),  # approx mem 30 GiB
                (64, 1, 5, [30 * 10**6]),  # approx mem 45 GiB

                # k_down = 1:
                (64, 2, 1, [2 * 10**5, 4 * 10**5]),  # approximated by 38785-bit k=1 XOR Arbiter PUF, approx mem 14.4 GiB
                #(64, 3, 1, [4 * 10**5]),  # approximated by 643161-bit k=1 XOR Arbiter PUF, approx mem 240 GiB
            ]
            for N in Ns
            for attack in [
                InterposePUFLRAttackExperiment
            ]
        ]

    def plot(self):
        data = self.experimenter.results
        data = data[data['n'] == 64]
        data['measured_time_h'] = data.apply(lambda row: row['measured_time'] / 60**2, axis=1)

        f, _ = subplots(1, 2, figsize=(8, 4))
        heatmap(
            data=data.groupby(['k_up', 'k_down'], as_index=True).max().reset_index().pivot(
                'k_down', 'k_up', 'accuracy'),
            annot=True,
            fmt='.2f',
            vmin=.5,
            vmax=1,
            ax=f.axes[0]
        )
        heatmap(
            data=data.groupby(['k_up', 'k_down'], as_index=True).max().reset_index().pivot(
                'k_down', 'k_up', 'N'),
            annot=True,
            fmt=',.0f',
            vmin=1,
            vmax=data['N'].max(),
            ax=f.axes[1],
            cmap=cm.rocket_r,
        )
        # heatmap(
        #     data=data.groupby(['k_up', 'k_down'], as_index=True).mean().reset_index().pivot(
        #         'k_down', 'k_up', 'measured_time_h'),
        #     annot=True,
        #     fmt='.2f',
        #     vmin=1,
        #     vmax=data['measured_time_h'].max(),
        #     ax=f.axes[2],
        #     cmap=cm.rocket_r,
        # )
        f.subplots_adjust(wspace=.2, top=.8)
        f.suptitle('Accuracy, Training Set Size, and Training Time of Training a\n'
                   'Chow-Parameter-based Approximation of 64-bit (k_up,k_down)-Interpose PUF ')
        f.savefig(f'figures/{self.name()}.pdf')
        f.savefig(f'figures/{self.name()}.png')
