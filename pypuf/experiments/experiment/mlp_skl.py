"""
This module provides an Experiment that generates a PUF with corresponding training data and passes them to a
(Scikit-learn) Multilayer Perceptron Learner.
"""

from os import getpid
from typing import NamedTuple, Iterable
from uuid import UUID

from numpy.random.mtrand import RandomState

from pypuf import tools
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.neural_networks.mlp_skl import MultiLayerPerceptronScikitLearn
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF


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
    k_down: int
    N: int
    validation_frac: float
    combiner: str
    preprocessing: str
    layers: Iterable[int]
    activation: str
    domain_in: int
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
    domain_out: int
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
        progress_log_name = None if not progress_log_prefix else \
            '{}_MLP_skl_{}'.format(progress_log_prefix, parameters.experiment_id)
        super().__init__(progress_log_name=progress_log_name, parameters=parameters)
        self.training_set = None
        self.simulation = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.simulation = InterposePUF(
            n=self.parameters.n,
            k_down=self.parameters.k_down,
            k_up=self.parameters.k_up,
            seed=self.parameters.seed_simulation,
            transform='atf',
        )
        self.training_set = tools.TrainingSet(
            instance=self.simulation,
            N=self.parameters.N,
            random_instance=RandomState(seed=self.parameters.seed_challenges),
        )
        self.learner = MultiLayerPerceptronScikitLearn(
            n=self.parameters.n,
            k=self.parameters.k,
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
        )
        self.learner.prepare()

    def run(self):
        """
        Execute the learning process.
        """
        self.model = self.learner.learn()

    def analyze(self):
        """
        Calculate statistics of the trained model and write them into a result log file.
        """
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
            domain_out=0,
            measured_time=self.measured_time,
            iterations=self.learner.nn.n_iter_,
            accuracy=accuracy,
            loss_curve=[round(loss, 3) for loss in self.learner.nn.loss_curve_],
            accuracy_curve=[round(accuracy, 3) for accuracy in self.learner.accuracy_curve],
            max_memory=self.max_memory(),
        )
