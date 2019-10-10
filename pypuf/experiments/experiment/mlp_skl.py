"""
This module provides an Experiment that generates a PUF with corresponding training data and passes them to a
(Scikit-learn) Multilayer Perceptron Learner.
"""

from os import getpid
from typing import NamedTuple, Iterable
from uuid import UUID

from numpy.random import RandomState
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.neural_networks.mlp_skl import MultiLayerPerceptronScikitLearn
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


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
    loss: str
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
    NUM_ACCURACY = 10000
    INTERPOSE = False

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
        self.INTERPOSE = self.parameters.transformation.startswith('interpose')
        if not self.INTERPOSE:
            self.simulation = LTFArray(
                weight_array=LTFArray.normal_weights(
                    n=self.parameters.n,
                    k=self.parameters.k,
                    random_instance=RandomState(seed=self.parameters.seed_simulation),
                ),
                transform=self.parameters.transformation,
                combiner=self.parameters.combiner,
            )
        else:
            params = self.parameters.transformation.split(sep=' ')
            params_dict = {}
            if len(params) > 1:
                for p in params[1:]:
                    p = p.split('=')
                    params_dict[p[0]] = p[1]
            n = int(params_dict['n']) if 'n' in params_dict.keys() else self.parameters.n
            k_down = int(params_dict['k_down']) if 'k_down' in params_dict.keys() else self.parameters.k
            k_up = int(params_dict['k_up']) if 'k_up' in params_dict.keys() else self.parameters.k
            interpose_pos = int(params_dict['interpose_pos']) if 'interpose_pos' in params_dict.keys() \
                                                                 and params_dict['interpose_pos'] != 'None' else None
            seed = params_dict['seed'] if 'seed' in params_dict.keys() else self.parameters.seed_simulation
            seed = int(seed) if seed != 'None' else None
            transform = params_dict['transform'] if 'transform' in params_dict.keys() else None
            noisiness = float(params_dict['noisiness']) if 'noisiness' in params_dict.keys() else 0
            noise_seed = params_dict['noise_seed'] if 'noise_seed' in params_dict.keys() \
                else self.parameters.seed_simulation
            noise_seed = int(noise_seed) if noise_seed != 'None' else None
            self.simulation = InterposePUF(
                n=n,
                k_down=k_down,
                k_up=k_up,
                interpose_pos=interpose_pos,
                seed=seed,
                transform=transform,
                noisiness=noisiness,
                noise_seed=noise_seed,
            )
        prng_challenges = RandomState(seed=self.parameters.seed_challenges)
        self.training_set = tools.TrainingSet(
            instance=self.simulation,
            N=self.parameters.N,
            random_instance=prng_challenges,
        )
        self.learner = MultiLayerPerceptronScikitLearn(
            n=self.parameters.n,
            k=self.parameters.k,
            training_set=self.training_set,
            validation_frac=self.parameters.validation_frac,
            transformation=self.simulation.down.transform if self.INTERPOSE else self.simulation.transform,
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
            num=min(self.NUM_ACCURACY, 2 ** self.parameters.n),
            random_instance=RandomState(seed=self.parameters.seed_distance),
        )
        return Result(
            name=self.NAME,
            experiment_id=self.id,
            pid=getpid(),
            domain_out=0,
            loss='log_loss',
            measured_time=self.measured_time,
            iterations=self.learner.nn.n_iter_,
            accuracy=accuracy,
            loss_curve=[round(loss, 3) for loss in self.learner.nn.loss_curve_],
            accuracy_curve=[round(accuracy, 3) for accuracy in self.learner.accuracy_curve],
            max_memory=self.max_memory(),
        )
