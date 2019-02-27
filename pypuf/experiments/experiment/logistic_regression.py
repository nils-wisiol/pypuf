"""
This module provides an experiment class which learns an instance of LTFArray simulation PUF with the logistic
regression learner.
"""
from os import getpid
from typing import NamedTuple
from uuid import UUID

from numpy.random import RandomState
from numpy.linalg import norm
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


class Parameters(NamedTuple):
    """
    Holds parameters for logistic regression experiments.
    """
    # Seeds
    seed_instance: int
    seed_model: int
    seed_challenge: int
    seed_distance: int

    # LTF array definition
    n: int
    k: int
    transformation: str
    combiner: str

    # Learning setup
    N: int
    mini_batch_size: int
    convergence_decimals: float
    shuffle: bool


class Result(NamedTuple):
    """
    Holds results from logistic regression experiments.
    """
    experiment_id: UUID
    pid: int
    iteration_count: int
    epoch_count: int
    gradient_step_count: int
    measured_time: float
    accuracy: float
    model: list


class ExperimentLogisticRegression(Experiment):
    """
    This Experiment uses the logistic regression learner on an LTFArray PUF simulation.
    """

    def __init__(self, progress_log_prefix, parameters):
        progress_log_name = None if progress_log_prefix is None else '%s.0x%x_0x%x_0_%i_%i_%i_%s_%s' % (
            progress_log_prefix,
            parameters.seed_model,
            parameters.seed_instance,
            parameters.n,
            parameters.k,
            parameters.N,
            parameters.transformation,
            parameters.combiner,
            )
        super().__init__(progress_log_name, parameters)
        self.instance = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Initializes the instance, the training set and the learner to then run the logistic regression
        with the given parameters.
        """
        self.instance = LTFArray(
            weight_array=LTFArray.normal_weights(
                self.parameters.n,
                self.parameters.k,
                random_instance=RandomState(seed=self.parameters.seed_instance)
            ),
            transform=self.parameters.transformation,
            combiner=self.parameters.combiner,
        )
        self.learner = LogisticRegression(
            tools.TrainingSet(
                instance=self.instance,
                N=self.parameters.N,
                random_instance=RandomState(self.parameters.seed_challenge)
            ),
            self.parameters.n,
            self.parameters.k,
            transformation=self.instance.transform,
            combiner=self.instance.combiner,
            weights_prng=RandomState(seed=self.parameters.seed_model),
            logger=self.progress_logger,
            minibatch_size=self.parameters.mini_batch_size,
            convergance_decimals=self.parameters.convergence_decimals or 2,
            shuffle=self.parameters.shuffle,
        )

    def run(self):
        """
        Runs the learner
        """
        self.model = self.learner.learn()

    def analyze(self):
        """
        Analyzes the learned result.
        """
        assert self.model is not None
        accuracy = 1.0 - tools.approx_dist(
            self.instance,
            self.model,
            min(10000, 2 ** self.parameters.n),
            random_instance=RandomState(self.parameters.seed_distance),
        )

        return Result(
            experiment_id=self.id,
            pid=getpid(),
            iteration_count=self.learner.iteration_count,
            epoch_count=self.learner.epoch_count,
            gradient_step_count=self.learner.gradient_step_count,
            measured_time=self.measured_time,
            accuracy=accuracy,
            model=self.model.weight_array.flatten() / norm(self.model.weight_array.flatten()),
        )
