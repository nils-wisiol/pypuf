from os import getpid
from typing import NamedTuple, List
from uuid import UUID

from matplotlib.pyplot import close
from numpy import concatenate, zeros, array, array2string, pi, sqrt, ones, ndarray, average, empty, ceil
from numpy.random.mtrand import RandomState
from scipy.stats import pearsonr
from seaborn import catplot, axes_style

from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF, XORArbiterPUF
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.simulation.base import Simulation
from pypuf.studies.base import Study
from pypuf.tools import ChallengeResponseSet, TrainingSet, approx_dist, approx_dist_nonrandom, BIT_TYPE


class Parameters(NamedTuple):
    n: int
    k_up: int
    k_down: int
    N: int
    seed: int


class Result(NamedTuple):
    experiment_id: UUID
    measured_time: float
    pid: int
    max_memory: int
    accuracy: float
    accuracy_up: float
    accuracy_down: float
    accuracies: List[float]
    accuracies_up: List[float]
    accuracies_down: List[float]
    correlation_down_front: object
    correlation_down_back: object
    correlation_up: object
    training_set_up_accuracy: List[float]
    training_set_down_accuracy: List[float]
    rounds: int
    first_rounds: int


class SplitAttack(Experiment):

    simulation: InterposePUF
    training_set: ChallengeResponseSet
    test_set: ChallengeResponseSet
    model_down: LTFArray
    model_up: LTFArray
    model: Simulation
    n2: int
    training_set_up_accuracy: List[float]
    training_set_down_accuracy: List[float]
    accuracies: List[float]
    accuracies_up: List[float]
    accuracies_down: List[float]

    def __init__(self, progress_log_name, parameters):
        super().__init__(progress_log_name, parameters)
        assert self.parameters.n % 2 == 0, f'n must be multiple of 2, but was {self.parameters.n}'
        self.n2 = self.parameters.n // 2
        self.training_set_up_accuracy = []
        self.training_set_down_accuracy = []
        self.accuracies = []
        self.accuracies_up = []
        self.accuracies_down = []
        self.rounds = 0
        self.first_rounds = 0

    def prepare(self):
        self.simulation = InterposePUF(
            n=self.parameters.n,
            k_down=self.parameters.k_down,
            k_up=self.parameters.k_up,
            transform='atf',
            seed=self.parameters.seed,
        )
        self.training_set = TrainingSet(self.simulation, self.parameters.N, RandomState(self.parameters.seed))
        self.test_set = TrainingSet(self.simulation, 10**4, RandomState(self.parameters.seed + 1))

    def run(self):
        while True:
            self.progress_logger.debug('computing first down model')
            self.model_down = self._get_first_model_down()

            # attacker model accuracy
            test_set_accuracy = 1 - approx_dist_nonrandom(self.model_down, self.test_set)

            # analysis: initial total accuracy
            self.accuracies.append(1 - approx_dist(self.model_down, self.simulation, 10 ** 4, RandomState(1)))

            # convert to down model
            self.model_down.weight_array = self._interpose(self.model_down.weight_array, sqrt(2/pi), replace=False)
            self.model_down.n += 1

            # analysis: down model accuracy
            self.accuracies_down.append(1 - approx_dist(self.model_down, self.simulation.down, 10 ** 4, RandomState(1)))
            self.progress_logger.debug(f'initial accuracy down: {self.accuracies_down[-1]:.2f} total: {self.accuracies[-1]:.2f}')

            # first model good?
            self.first_rounds += 1
            if not (.45 <= test_set_accuracy <= .55) or self.first_rounds > 10:
                break

        # early stop?
        if .45 <= test_set_accuracy <= .55:
            self.progress_logger.debug('initial down model is bad, we give up')
            self.model_up = XORArbiterPUF(
                n=self.parameters.n,
                k=self.parameters.k_up,
                seed=1,
                transform='atf',
            )
            self.accuracies_up.append(-1)
            self._update_model()
            self.rounds = 0
            return

        # iteratively train up, down, up, down, ...
        while True:
            def done():
                return self.rounds > 5 or 1 - approx_dist_nonrandom(self.model, self.test_set) >= .95

            self.model_up = self._get_model_up()
            self._update_model()
            if done():
                break

            self.model_down = self._get_next_model_down()
            self._update_model()
            self.rounds += 1
            if done():
                break

    def _update_model(self):
        self.model = InterposePUF(
            n=self.parameters.n,
            k_down=self.parameters.k_down,
            k_up=self.parameters.k_up,
            transform='atf',
        )
        self.model.up = self.model_up
        self.model.down = self.model_down

        # analysis: model accuracy
        self.accuracies.append(1 - approx_dist(self.model, self.simulation, 10**4, RandomState(1)))
        self.progress_logger.debug(f'current accuracy up: {self.accuracies_up[-1]:.2f}, '
                                   f'down: {self.accuracies_down[-1]:.2f}, total: {self.accuracies[-1]}')

    def _get_first_model_down(self):
        self.progress_logger.debug('initially training down model')
        learner = LogisticRegression(
            t_set=self.training_set,
            n=self.parameters.n,
            k=self.parameters.k_down,
            transformation=self.simulation.down.transform,
            weights_prng=RandomState(self.parameters.seed + 271828 + self.first_rounds),
            logger=self.progress_logger,
        )
        return learner.learn()

    def _get_next_model_down(self):
        # create a training set for the lower PUF, based on the upper layer model
        self.progress_logger.debug(f'copying challenges of size {self.training_set.challenges.nbytes / 1024**3:.2f}GiB')
        challenges = self.training_set.challenges[:, :]
        responses = self.model_up.eval(challenges)
        challenges = self._interpose(challenges, responses)
        training_set = ChallengeResponseSet(challenges, self.training_set.responses)

        # analysis: training set accuracy
        self.training_set_down_accuracy.append(average(
            self.simulation.down.eval(training_set.challenges) == training_set.responses,
        ))
        self.progress_logger.debug(f'new down training set accuracy: {self.training_set_down_accuracy[-1]:.2f}')

        # model training
        self.progress_logger.debug('re-training down model')
        learner = LogisticRegression(
            t_set=training_set,
            n=self.parameters.n + 1,
            k=self.parameters.k_down,
            transformation=self.simulation.down.transform,
            weights_prng=RandomState(0),  # overwritten below
            logger=self.progress_logger,
        )
        model_down = learner.learn(init_weight_array=self.model_down.weight_array)
        self.accuracies_down.append(1 - approx_dist(model_down, self.simulation.down, 10 ** 4, RandomState(1)))
        self.progress_logger.debug(f'new down model accuracy: {self.accuracies_down[-1]:.2f}')
        
        return model_down

    def _get_model_up(self):
        # create a training set for the upper PUF, based on the lower layer model
        N, n = self.parameters.N, self.parameters.n
        challenges = self.training_set.challenges
        responses = self.training_set.responses

        self.progress_logger.debug('creating training set for upper layer')
        block_size = 10**6

        # the max. number of challenges we select must be set up in advance to benefit from
        # numpy memory and cpu efficiency. While this number is non-deterministic and depends
        # on the given instance and training set, we expect it to be about N/2. To get a
        # reasonable upper bound, we add about 500MB of margin and round up to the next
        # multiple of block_size.
        N_selected = int((N/2 + (500 * 1024**2 / n)) // block_size + 1) * block_size
        selected_challenges = empty(shape=(N_selected, n), dtype=BIT_TYPE)
        self.progress_logger.debug(f'setting the max number of challenges for the upper training set to '
                                   f'{N_selected}, using {selected_challenges.nbytes / 1024**3:.2f}GiB, about '
                                   f'{N_selected / N * 100:.2f}% of all challenges')
        selected_responses = empty(shape=(N_selected,), dtype=BIT_TYPE)
        filled = 0
        for idx in range(int(ceil(N / block_size))):
            # carve out a block to work on, max. size block_size, true size block_len
            block_slice = slice(idx * block_size, (idx + 1) * block_size)
            block_challenges = challenges[block_slice]
            block_responses = responses[block_slice]
            block_len = len(block_challenges)

            # create extended challenges (number: 2 * block_len)
            challenges_p1 = self._interpose(block_challenges, +ones(shape=(block_len, 1)))
            challenges_m1 = self._interpose(block_challenges, -ones(shape=(block_len, 1)))

            # evaluate extended challenges
            responses_p1 = self.model_down.eval(challenges_p1)
            responses_m1 = self.model_down.eval(challenges_m1)

            # identify challenges that depend on the interpose bit, i.e. yield unequal response
            # for unequal interpose bit
            responses_unequal = responses_p1 != responses_m1
            unequal = sum(responses_unequal)

            # copy all these challenges (without interpose bit) to selected_challenges
            block_new = slice(
                filled,
                filled + unequal
            )
            selected_challenges[block_new] = block_challenges[responses_unequal]

            # to create the training set for the upper layer, we use the interpose bit that yielded
            # the response that matched our training set as the response of the upper layer, i.e.
            # select +1 if +1 as interpose bit yielded the correct response, or
            # select -1 if -1 as interpose bit yielded the correct response.
            # Let r be the response bit as recorded in the training set, rp1 the response bit
            # of the challenge with 1 as interpose bit. Then
            # selected_response = +1 if rp1 == r else -1.
            # This is the same as selected_response = rp1 * r.
            # We apply the product using numpy for all challenges at a time.
            selected_responses[block_new] = responses_p1[responses_unequal] * block_responses[responses_unequal]

            filled += unequal
            self.progress_logger.debug(f'wrote selected {{challenges, responses}} from {filled} to '
                                       f'{filled + unequal}')

        # cut off selected_challenges and selected_responses to the correct size
        training_set_up = ChallengeResponseSet(selected_challenges[:filled], selected_responses[:filled])
        self.progress_logger.debug(f'training set for upper layer created, size '
                                   f'{training_set_up.challenges.nbytes / 1024**3}GiB')

        # analysis: training set accuracy
        self.training_set_up_accuracy.append(average(
            self.simulation.up.eval(training_set_up.challenges) == training_set_up.responses
        ))
        self.progress_logger.debug(f'new up training set accuracy: {self.training_set_up_accuracy[-1]:.2f} size: {len(selected_challenges)}')

        # train the upper model
        self.progress_logger.debug('(re)training up model')
        learner = LogisticRegression(
            t_set=training_set_up,
            n=self.parameters.n,
            k=self.parameters.k_up,
            transformation=self.simulation.up.transform,
            weights_prng=RandomState(self.parameters.seed + 43),
            logger=self.progress_logger,
        )
        model_up = learner.learn() if not getattr(self, 'model_up', None) else learner.learn(init_weight_array=self.model_up.weight_array)
        self.accuracies_up.append(1 - approx_dist(model_up, self.simulation.up, 10 ** 4, RandomState(1)))
        self.progress_logger.debug(f'new up model accuracy: {self.accuracies_up[-1]:.2f}')

        return model_up

    @staticmethod
    def _weight_correlation(x, y):
        return array2string(
            array([[pearsonr(x[i, :], y[j, :])[0] for i in range(x.shape[0])] for j in range(y.shape[0])]),
            precision=1,
            floatmode='fixed',
            suppress_small=True,
        )

    def analyze(self):
        return Result(
            experiment_id=self.id,
            measured_time=self.measured_time,
            pid=getpid(),
            max_memory=self.max_memory(),
            accuracy=1 - approx_dist(self.simulation, self.model, 10**4, RandomState(31415)),
            accuracy_up=1 - approx_dist(self.simulation.up, self.model_up, 10**4, RandomState(31416)),
            accuracy_down=1 - approx_dist(self.simulation.down, self.model_down, 10**4, RandomState(31417)),
            correlation_up=self._weight_correlation(
                self.simulation.up.weight_array,
                self.model_up.weight_array,
            ),
            correlation_down_front=self._weight_correlation(
                self.simulation.down.weight_array[:, :self.n2],
                self.model_down.weight_array[:, :self.n2],
            ),
            correlation_down_back=self._weight_correlation(
                self.simulation.down.weight_array[:, self.n2+1:],
                self.model_down.weight_array[:, self.n2+1:],
            ),
            training_set_up_accuracy=self.training_set_up_accuracy,
            training_set_down_accuracy=self.training_set_down_accuracy,
            accuracies=self.accuracies,
            accuracies_up=self.accuracies_up,
            accuracies_down=self.accuracies_down,
            rounds=self.rounds,
            first_rounds=self.first_rounds,
        )

    def _interpose(self, challenges, bits, replace=None):
        n = self.parameters.n
        if replace is None:
            replace = challenges.shape[1] == n + 1
        if replace:
            return self._interpose_replace(challenges, bits)
        else:
            return self._interpose_insert(challenges, bits)

    def _interpose_replace(self, challenges, bits):
        challenges = challenges[:, :]
        if isinstance(bits, ndarray):
            challenges[:, self.n2] = bits.reshape((challenges.shape[0],))
            return challenges
        else:
            challenges[:, self.n2] = zeros(shape=(challenges.shape[0],)) + bits
            return challenges

    def _interpose_insert(self, challenges, bits):
        if isinstance(bits, ndarray):
            N = challenges.shape[0]
            return concatenate((challenges[:, :self.n2], bits.reshape(N, 1), challenges[:, self.n2:]), axis=1)
        else:
            return concatenate(
                (
                    challenges[:, :self.n2],
                    zeros(shape=(challenges.shape[0], 1)) + bits,
                    challenges[:, self.n2:]
                ), axis=1
            )


class SplitAttackStudy(Study):

    SHUFFLE = True

    def experiments(self):
        return [
            SplitAttack(
                progress_log_name=f'{self.name()}-n={n}-k_up={k_up}-k_down={k_down}-N={N}-seed={seed}',
                parameters=Parameters(
                    n=n, k_up=k_up, k_down=k_down, N=N, seed=seed,
                )
            )
            for n in [64]
            for k_up, k_down, N in [
                #(1, 1, 10000),
                #(2, 2, 5),
                #(3, 3, 80000),
                #(4, 4, 100000),
                #(5, 5, 600000),
                # (6, 6, 2000000),
                # (6, 6, 5000000),
                # (7, 7, 8000000),
                # (7, 7, 20000000),
                # (7, 7, 40000000),
                (8, 8, 80000000),
                (8, 8, 120000000),
                (9, 9, 120000000),
                (9, 9, 150000000),
                # (1, 4, 200000),
                # (1, 4, 400000),
                # (1, 5, 400000),
                # (1, 5, 1000000),
                # (1, 6, 2000000),
                # (1, 6, 4000000),
                # (1, 7, 4000000),
                # (1, 7, 8000000),
                # (1, 7, 12000000),
                # (1, 7, 20000000),
                (1, 8, 40000000),
                (1, 8, 80000000),
                (1, 8, 120000000),
                (1, 9, 100000000),
                (1, 9, 120000000),
                (1, 9, 150000000),
            ]
            for seed in range(100)
        ]

    def plot(self):
        data = self.experimenter.results
        data['max_memory_gb'] = data.apply(lambda row: row['max_memory'] / 1024**3, axis=1)
        data['Ne6'] = data.apply(lambda row: row['N'] / 1e6, axis=1)
        data['size'] = data.apply(lambda row: '(%i,%i)' % (int(row['k_up']), int(row['k_down'])), axis=1)
        data['measured_time'] = data.apply(lambda row: round(row['measured_time']), axis=1)
        data = data.sort_values(['size'])
        with axes_style('whitegrid'):
            f = catplot(
                data=data,
                x='Ne6',
                y='accuracy',
                row='size',
                hue='measured_time',
                kind='swarm',
                aspect=3,
                height=2,
                legend=False,
            )
            for ax in f.axes.flatten():
                ax.set(ylim=(.45, 1))
            f.savefig(f'figures/{self.name()}.pdf')
            close(f.fig)
