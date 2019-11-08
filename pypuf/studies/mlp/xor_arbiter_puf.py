"""
This module describes a study that defines a set of experiments in order to find good hyperparameters and to examine
the quality of Deep Learning based modeling attacks on XOR Arbiter PUFs. Furthermore, some plots are defined to
visualize the experiment's results.
The Deep Learning technique used here is the Optimization technique Adam for a Stochastic Gradient Descent on a Feed-
Forward Neural Network architecture called Multilayer Perceptron (MLP). Implementations of the MLP and Adam are
used from Scikit-Learn and Tensorflow, respectively.
"""
from os import getpid
from typing import NamedTuple, Iterable
from uuid import UUID, uuid4

from numpy.random.mtrand import RandomState
from seaborn import catplot

from pypuf import tools
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.neural_networks.mlp_skl import MultiLayerPerceptronScikitLearn
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF
from pypuf.studies.base import Study


class Parameters(NamedTuple):
    """
    Define all input parameters for the Experiment.
    """
    seed_simulation: int
    seed_challenges: int
    seed_model: int
    seed_distance: int
    n: int
    k: int
    N: int
    validation_frac: float
    transformation: str
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
        self.simulation = XORArbiterPUF(
            n=self.parameters.n,
            k=self.parameters.k,
            seed=self.parameters.seed_simulation,
            transform=self.parameters.transformation,
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
            k=self.parameters.k,
            training_set=self.training_set,
            validation_frac=self.parameters.validation_frac,
            transformation=self.simulation.transform,
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
            num=10**4,
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


class MLPDiversePUFsStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """

    COMBINER = 'xor'
    ACTIVATION = 'relu'
    ITERATION_LIMIT = 40
    LOSS = 'log_loss'
    DOMAIN_IN = -1
    PATIENCE = 4
    TOLERANCE = 0.0025
    PENALTY = 0.0002
    BETA_1 = 0.9
    BETA_2 = 0.999
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False
    SHUFFLE = True

    EXPERIMENTS = []

    TRANSFORMATIONS = ['atf']

    PREPROCESSINGS = ['short']

    SIZES = {
        (64, 4): [0.4e6],
        (64, 5): [0.8e6],
        (64, 6): [2e6],
        (64, 7): [5e6],
        (64, 8): [30e6],
    }

    SAMPLES_PER_POINT = {
        (64, 4): 100,
        (64, 5): 100,
        (64, 6): 100,
        (64, 7): 100,
        (64, 8): 100,
    }

    LEARNING_RATES = {
        (64, 4): [0.0025],
        (64, 5): [0.0025],
        (64, 6): [0.0055],
        (64, 7): [0.002],
        (64, 8): [0.001],
    }

    LAYERS = {
        (64, 4): [[2 ** 4, 2 ** 4, 2 ** 4]],
        (64, 5): [[2 ** 5, 2 ** 5, 2 ** 5]],
        (64, 6): [[2 ** 6, 2 ** 6, 2 ** 6]],
        (64, 7): [[2 ** 7, 2 ** 7, 2 ** 7]],
        (64, 8): [[2 ** 8, 2 ** 8, 2 ** 8]],
    }

    BATCH_SIZES = {
        (64, 4): [1000],
        (64, 5): [1000],
        (64, 6): [10000],
        (64, 7): [10000],
        (64, 8): [10000],
    }

    def experiments(self):
        """
        Generate an experiment for every parameter combination corresponding to the definitions above.
        For each combination of (size, samples_per_point, learning_rate) different random seeds are used.
        """
        for c1, (n, k) in enumerate(self.SIZES.keys()):
            for N in self.SIZES[(n, k)]:
                validation_frac = max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N
                for c2 in range(self.SAMPLES_PER_POINT[(n, k)]):
                    for c3, learning_rate in enumerate(self.LEARNING_RATES[(n, k)]):
                        cycle = c1 * (self.SAMPLES_PER_POINT[(n, k)] * len(self.LEARNING_RATES[(n, k)])) \
                                + c2 * len(self.LEARNING_RATES[(n, k)]) + c3
                        for preprocessing in self.PREPROCESSINGS:
                            for layers in self.LAYERS[(n, k)]:
                                for transformation in self.TRANSFORMATIONS:
                                    for batch_size in self.BATCH_SIZES[(n, k)]:
                                        self.EXPERIMENTS.append(
                                            ExperimentMLPScikitLearn(
                                                progress_log_prefix=self.name(),
                                                parameters=Parameters(
                                                    seed_simulation=0x3 + cycle,
                                                    seed_challenges=0x1415 + cycle,
                                                    seed_model=0x9265 + cycle,
                                                    seed_distance=0x3589 + cycle,
                                                    n=n,
                                                    k=k,
                                                    N=int(N),
                                                    validation_frac=validation_frac,
                                                    transformation=transformation,
                                                    combiner=self.COMBINER,
                                                    preprocessing=preprocessing,
                                                    layers=layers,
                                                    activation=self.ACTIVATION,
                                                    domain_in=self.DOMAIN_IN,
                                                    learning_rate=learning_rate,
                                                    penalty=self.PENALTY,
                                                    beta_1=self.BETA_1,
                                                    beta_2=self.BETA_2,
                                                    tolerance=self.TOLERANCE,
                                                    patience=self.PATIENCE,
                                                    iteration_limit=self.ITERATION_LIMIT,
                                                    batch_size=batch_size,
                                                    print_learning=self.PRINT_LEARNING,
                                                    domain_out=-1,
                                                    loss='log_loss',
                                                )
                                            )
                                        )
        return self.EXPERIMENTS

    def plot(self):
        data = self.experimenter.results
        data['Ne6'] = data.apply(lambda row: row['N'] / 10e6, axis=1)
        f = catplot(
            data=data,
            x='Ne6',
            y='accuracy',
            col='k',
            row='transformation',
            hue='layers',
            aspect=.4,
            height=10,
        )
        f.fig.savefig(f'figures/{self.name()}.pdf')
