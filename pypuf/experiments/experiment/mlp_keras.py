from os import getpid
from uuid import UUID
from typing import NamedTuple, List

from numpy.random.mtrand import RandomState
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.neural_networks.mlp import MultiLayerPerceptron
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


class Parameters(NamedTuple):
    layers: List[int]
    learning_rate: float
    beta_1: float
    beta_2: float
    N: int
    n: int
    k: int
    transformation: str
    combiner: str
    preprocess: bool
    batch_size: int
    iteration_limit: int
    print_keras: bool
    seed_simulation: int
    seed_challenges: int
    seed_model: int
    seed_accuracy: int


class Result(NamedTuple):
    experiment_id: UUID
    pid: int
    accuracy: float
    accuracy_val: float
    accuracy_train: float
    epochs: int
    measured_time: float
    layers: str
    learning_rate: float
    beta_1: float
    beta_2: float


class ExperimentMLP(Experiment):
    """
    This Experiment uses the MLP learner on an LTFArray PUF simulation.
    """

    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    NUM_ACCURACY = 10000

    def __init__(self, progress_log_prefix, parameters):
        progress_log_name = None if not progress_log_prefix else '{}_MLP_0x{}_0x{}_0_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            progress_log_prefix,
            parameters.seed_model,
            parameters.seed_simulation,
            parameters.n,
            parameters.k,
            parameters.N,
            parameters.layers,
            parameters.learning_rate,
            parameters.beta_1,
            parameters.beta_2,
            parameters.transformation,
            parameters.combiner,
        )
        super().__init__(progress_log_name, parameters)
        self.training_set = None
        self.validation_set = None
        self.simulation = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.simulation = LTFArray(
            weight_array=LTFArray.normal_weights(
                self.parameters.n,
                self.parameters.k,
                random_instance=RandomState(seed=self.parameters.seed_simulation)
            ),
            transform=self.parameters.transformation,
            combiner=self.parameters.combiner
        )
        prng_challenges = RandomState(seed=self.parameters.seed_challenges)
        N_val = max(min(self.parameters.N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL)
        N_train = self.parameters.N - N_val
        self.training_set = tools.TrainingSet(self.simulation, N_train, prng_challenges)
        self.validation_set = tools.TrainingSet(self.simulation, N_val, prng_challenges)
        self.learner = MultiLayerPerceptron(
            layers=self.parameters.layers,
            n=self.parameters.n,
            k=self.parameters.k,
            training_set=self.training_set,
            validation_set=self.validation_set,
            transformation=self.simulation.transform if self.parameters.preprocess else None,
            learning_rate=self.parameters.learning_rate,
            beta_1=self.parameters.beta_1,
            beta_2=self.parameters.beta_2,
            checkpoint_name=self.id,
            print_keras=self.parameters.print_keras,
            termination_threshold=1.0,
            iteration_limit=self.parameters.iteration_limit,
            batch_size=self.parameters.batch_size,
            seed_model=self.parameters.seed_model
        )
        self.learner.prepare()

    def run(self):
        """
        Train the learner.
        """
        self.model = self.learner.learn()

    def analyze(self):
        """
        Calculate statistics of the trained model and write them into a result log file.
        """
        assert self.model is not None
        accuracy = 1.0 - tools.approx_dist(
                self.simulation,
                self.model,
                min(self.NUM_ACCURACY, 2 ** self.parameters.n),
                random_instance=RandomState(seed=self.parameters.seed_accuracy)
        )
        return Result(
            experiment_id=self.id,
            pid=getpid(),
            accuracy=accuracy,
            accuracy_val=self.learner.history.history['val_pypuf_accuracy'][-1],
            accuracy_train=self.learner.history.history['pypuf_accuracy'][-1],
            epochs=self.learner.history.epoch[-1],
            measured_time=self.measured_time,
            layers=str(self.parameters.layers),
            learning_rate=self.parameters.learning_rate,
            beta_1=self.parameters.beta_1,
            beta_2=self.parameters.beta_2,
        )
