from os import getpid
from typing import NamedTuple, Iterable, Tuple
from uuid import UUID

from numpy.random import RandomState
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.mlp import MLP
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


class Parameters(NamedTuple):
    seed_instance: int
    seed_model: int
    seed_distance: int
    seed_accuracy: int
    n: int
    k: int
    N: int
    transformation: str
    combiner: str
    batch_size: int
    iteration_limit: int
    initial_model_sigma: float
    learning_rate: float
    activation: str
    hidden_layer_sizes: Iterable[int]
    tol: float
    n_iter_no_change: int


class Result(NamedTuple):
    experiment_id: UUID
    pid: int
    measured_time: float
    iterations: int
    accuracy: float
    loss_curve: Iterable[float]


class ExperimentMLP(Experiment):
    """
    This Experiment uses the MLP learner on an LTFArray PUF simulation.
    """

    def __init__(self, log_name, parameters):
        super().__init__(
            progress_log_name=None if log_name is None else '%s_MLP_0x%x_0x%x_0_%i_%i_%i_%s_%s' % (
                log_name,
                parameters.seed_model,
                parameters.seed_instance,
                parameters.n,
                parameters.k,
                parameters.N,
                parameters.transformation,
                parameters.combiner,
            ),
            parameters=parameters
        )
        self.simulation = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.simulation = LTFArray(
            weight_array=LTFArray.normal_weights(self.parameters.n, self.parameters.k, 0,
                                                 self.parameters.initial_model_sigma,
                                                 random_instance=RandomState(seed=self.parameters.seed_instance)),
            transform=self.parameters.transformation,
            combiner=self.parameters.combiner,
        )
        self.learner = MLP(
            N=self.parameters.N,
            n=self.parameters.n,
            k=self.parameters.k,
            simulation=self.simulation,
            seed_distance=self.parameters.seed_distance,
            seed_model=self.parameters.seed_model,
            batch_size=self.parameters.batch_size,
            iteration_limit=self.parameters.iteration_limit,
            learning_rate=self.parameters.learning_rate,
            activation=self.parameters.activation,
            hidden_layer_sizes=self.parameters.hidden_layer_sizes,
            tol=self.parameters.tol,
            n_iter_no_change=self.parameters.n_iter_no_change,
        )
        self.learner.prepare()

    def run(self):
        """
        Runs the learner
        """
        self.model = self.learner.learn()

    def analyze(self):
        """
        Analyzes the learned result.
        """
        return Result(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            iterations=self.learner.clf.n_iter_,
            accuracy=1.0 - tools.approx_dist(
                self.simulation,
                self.model,
                min(10000, 2 ** self.parameters.n),
                random_instance=RandomState(seed=self.parameters.seed_accuracy)
            ),
            loss_curve=self.learner.clf.loss_curve_,
        )
