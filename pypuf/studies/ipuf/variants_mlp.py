"""
This module describes a study that defines a set of experiments in order to examine the quality of Deep Learning based
modeling attacks on XOR Arbiter PUFs. Furthermore, some plots are defined to visualize the experiment's results.
The Deep Learning technique used here is the Feed-Forward Neural Network architecture called Multilayer Perceptron (MLP)
[1] that is applied with the optimization technique Adam [2] for a Stochastic Gradient Descent. Implementations of the
MLP and Adam are used from Scikit-Learn [3].

References:
[1]  F. Rosenblatt,         "The Perceptron: A Probabilistic Model for Information Storage and Organization in the
                            Brain.", Psychological Review, volume 65, pp. 386-408, 1958.
[2]  D. Kingma and J. Ba,   “Adam: A Method for Stochastic Optimization”, arXiv:1412.6980, 2014.
[3]  F., Pedregosa et al.,  "Scikit-learn: Machine learning in Python", Journal of Machine Learning Research, volume 12,
                            pp. 2825-2830, 2011.
                            https://scikit-learn.org
"""
from os import getpid
from typing import NamedTuple, Iterable
from uuid import UUID
from uuid import uuid4

from matplotlib.pyplot import close
from numpy import concatenate
from numpy.core._multiarray_umath import ndarray
from numpy.random.mtrand import RandomState
from seaborn import catplot, axes_style

from pypuf import tools
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.neural_networks.mlp_skl import MultiLayerPerceptronScikitLearn
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF, XORArbiterPUF
from pypuf.simulation.base import Simulation
from pypuf.studies.base import Study


class Interpose3PUF(Simulation):

    def __init__(self, n: int, k_up: int, k_middle: int, k_down: int, seed: int) -> None:
        seeds = RandomState(seed)
        self.n = n
        self.k_up, self.k_middle, self.k_down = k_up, k_middle, k_down
        self.up = XORArbiterPUF(n, k_up, seed=seeds.randint(0, 2 ** 32))
        self.middle = XORArbiterPUF(n, k_up, seed=seeds.randint(0, 2 ** 32))
        self.down = XORArbiterPUF(n, k_up, seed=seeds.randint(0, 2 ** 32))
        self.interpose_pos = n // 2

    def challenge_length(self) -> int:
        return self.up.challenge_length()

    def response_length(self) -> int:
        return self.down.response_length()

    def _interpose(self, challenges, bits):
        ipos = self.interpose_pos
        return concatenate(
            challenges[:, :ipos], bits, challenges[:, ipos:],
            axis=1,
        )

    def eval(self, challenges: ndarray) -> ndarray:
        return self.down.eval(self._interpose(
            challenges, self.middle.eval(self._interpose(challenges, self.up.eval(challenges)))))


class Parameters(NamedTuple):
    """
    Define all input parameters for the Experiment.
    """
    seed_simulation: int
    seed_challenges: int
    seed_model: int
    seed_distance: int
    n: int
    k_up: int
    k_middle: int
    k_down: int
    N: int
    validation_frac: float
    combiner: str
    preprocessing: str
    layers: Iterable[int]
    activation: str
    loss: str
    domain_in: int
    domain_out: int
    learning_rate: float
    penalty: float
    beta_1: float
    beta_2: float
    tolerance: float
    patience: int
    iteration_limit: int
    batch_size: int
    print_learning: bool


class Result(NamedTuple):
    """
    Define all parameters to be documented within the result file that are not included in the input parameters.
    """
    name: str
    experiment_id: UUID
    pid: int
    measured_time: float
    iterations: int
    accuracy: float
    loss_curve: Iterable[float]
    accuracy_curve: Iterable[float]
    max_memory: int


class ExperimentMLPScikitLearn(Experiment):
    """
    This Experiment uses the Scikit-learn implementation of the Multilayer Perceptron Learner.
    """

    NAME = 'Multilayer Perceptron (scikit-learn)'

    def __init__(self, progress_log_prefix, parameters):
        self.id = uuid4()
        progress_log_name = None if not progress_log_prefix else \
            '{}_{}'.format(progress_log_prefix, self.id)
        super().__init__(progress_log_name=progress_log_name, parameters=parameters)
        self.training_set = None
        self.simulation = None
        self.learner = None
        self.model = None
        assert self.parameters.domain_out == -1, 'domain_out other than -1 is not supported, sorry.'
        assert self.parameters.loss == 'log_loss', 'loss other than log_loss is not supported, sorry.'

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.progress_logger.debug('Setting up simulation')
        self.simulation = InterposePUF(
            n=self.parameters.n,
            k_up=self.parameters.k_up,
            k_down=self.parameters.k_down,
            seed=self.parameters.seed_simulation,
        )
        self.progress_logger.debug(f'Gathering training set with {self.parameters.N} examples')
        self.training_set = tools.TrainingSet(
            instance=self.simulation,
            N=self.parameters.N,
            random_instance=RandomState(seed=self.parameters.seed_challenges),
        )
        self.progress_logger.debug('Setting up learner')
        self.learner = MultiLayerPerceptronScikitLearn(
            n=self.parameters.n,
            k=self.parameters.k_down,
            training_set=self.training_set,
            validation_frac=self.parameters.validation_frac,
            transformation=self.simulation.down.transform,
            preprocessing=self.parameters.preprocessing,
            layers=self.parameters.layers,
            learning_rate=self.parameters.learning_rate,
            penalty=self.parameters.penalty,
            beta_1=self.parameters.beta_1,
            beta_2=self.parameters.beta_2,
            tolerance=self.parameters.tolerance,
            patience=self.parameters.patience,
            iteration_limit=self.parameters.iteration_limit,
            batch_size=self.parameters.batch_size,
            seed_model=self.parameters.seed_model,
            print_learning=self.parameters.print_learning,
            logger=self.progress_logger.debug,
        )
        self.learner.prepare()

    def run(self):
        """
        Execute the learning process.
        """
        self.progress_logger.debug('Starting learner')
        self.model = self.learner.learn()

    def analyze(self):
        """
        Calculate statistics of the trained model and write them into a result log file.
        """
        self.progress_logger.debug('Analyzing result')
        assert self.model is not None
        accuracy = 1.0 - tools.approx_dist(
            instance1=self.simulation,
            instance2=self.model,
            num=10 ** 4,
            random_instance=RandomState(seed=self.parameters.seed_distance),
        )
        return Result(
            name=self.NAME,
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            iterations=self.learner.nn.n_iter_,
            accuracy=accuracy,
            loss_curve=[round(loss, 3) for loss in self.learner.nn.loss_curve_],
            accuracy_curve=[round(accuracy, 3) for accuracy in self.learner.accuracy_curve],
            max_memory=self.max_memory(),
        )


class InterposeMLPStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """
    SHUFFLE = True

    ITERATION_LIMIT = 150
    PATIENCE = ITERATION_LIMIT
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False

    SIZES = {
        (1, 1, 1): (
            [16, 24, 32],
            [5 * 10 ** 5, 1 * 10 ** 6, 2 * 10 ** 6],
            [.0002, .0004],
        ),
        (1, 1, 2): (
            [16, 24, 32],
            [5 * 10 ** 5, 1 * 10 ** 6, 2 * 10 ** 6],
            [.0002, .0004],
        ),
        (2, 2, 2): (
            [16, 24, 32],
            [5 * 10**5, 1 * 10**6, 2 * 10**6],
            [.0002, .0004],
        )
    }

    def experiments(self):
        return [
            ExperimentMLPScikitLearn(
                progress_log_prefix=self.name(),
                parameters=Parameters(
                    seed_simulation=0x3 + seed,
                    seed_challenges=0x1415 + seed,
                    seed_model=0x9265 + seed,
                    seed_distance=0x3589 + seed,
                    n=n,
                    k_up=k_up,
                    k_middle=k_middle,
                    k_down=k_down,
                    N=N,
                    validation_frac=max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N,
                    combiner='xor',
                    preprocessing='short',
                    layers=[layer_size, layer_size, layer_size],
                    activation='relu',
                    domain_in=-1,
                    learning_rate=learning_rate,
                    penalty=0.0002,
                    beta_1=0.9,
                    beta_2=0.999,
                    tolerance=0.0025,
                    patience=self.PATIENCE,
                    iteration_limit=self.ITERATION_LIMIT,
                    batch_size=batch_size,
                    print_learning=self.PRINT_LEARNING,
                    loss='log_loss',
                    domain_out=-1,
                )
            )
            for seed in range(10)
            for batch_size in [10 ** 5]
            for n in [64]
            for (k_up, k_middle, k_down), (LS_set, N_set, LR_set) in self.SIZES.items()
            for layer_size in LS_set
            for N in N_set
            for learning_rate in LR_set
        ]

    def plot(self):
        data = self.experimenter.results
        data['size'] = data.apply(lambda row: f'({row["k_up"]:.0f}, {row["k_middle"]:.0f}, {row["k_down"]:.0f})', axis=1)
        data['max_memory_gb'] = data.apply(lambda row: row['max_memory'] / 1024 ** 3, axis=1)
        data['Ne6'] = data.apply(lambda row: row['N'] / 1e6, axis=1)
        data = data.sort_values(['size', 'layers'])

        with axes_style('whitegrid'):
            params = dict(
                data=data,
                x='Ne6',
                y='accuracy',
                row='size',
                kind='swarm',
                aspect=2,
                height=4,
            )
            for name, params_ind in {
                'layer': dict(hue='layers', hue_order=[str([2 ** s] * 3) for s in range(2, 10)]),
                'learning_rate': dict(hue='learning_rate'),
            }.items():
                f = catplot(**params, **params_ind)
                f.axes.flatten()[0].set(ylim=(.45,1.))
                f.savefig(f'figures/{self.name()}.{name}.pdf')
                close(f.fig)
