"""
This module provides an experiment class which learns an instance of LTFArray simulation PUF that uses
the lightweight-secure transform with the correlation attack learner.
"""
from os import getpid
from typing import NamedTuple, Tuple
from uuid import UUID

from numpy.random import RandomState
from numpy.linalg import norm
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.correlation_attack import CorrelationAttack
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import TrainingSet, approx_dist
from itertools import permutations
from scipy.stats import pearsonr


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

    # Learning setup
    N: int
    lr_iteration_limit: int
    mini_batch_size: int
    convergence_decimals: float
    shuffle: bool


class Result(NamedTuple):
    """
    Holds results from logistic regression experiments.
    """
    experiment_id: UUID
    pid: int
    measured_time: float

    # Results from the initial LR run
    initial_model: list
    initial_lr_iterations: int
    initial_accuracy: float

    # Correlation attack specifics

    # The correct permutation for the initial_model
    correct_permutation: Tuple

    # Best performing permutation found by the correlation attack
    best_permutation: Tuple

    # Number of permutations tried until best performing was found
    best_permutation_iteration: int

    # Total number of permutations tried
    total_permutation_iterations: int

    # Total number of logistic regression iterations
    total_lr_iterations: int

    # Final results
    model: list
    accuracy: float


class ExperimentCorrelationAttack(Experiment):
    """
    This Experiment uses the CorrelationAttack learner on an LTFArray PUF simulation.
    """

    def __init__(self, progress_log_prefix, parameters):
        super().__init__(
            progress_log_name='%s.0x%x_0x%x_0_%i_%i_%i_%s_%s' % (
                progress_log_prefix,
                parameters.seed_model,
                parameters.seed_instance,
                parameters.n,
                parameters.k,
                parameters.N,
                LTFArray.transform_lightweight_secure.__name__,
                LTFArray.combiner_xor.__name__,
            ),
            parameters=parameters
        )
        self.n = parameters.n
        self.k = parameters.k
        self.N = parameters.N
        self.lr_iteration_limit = parameters.lr_iteration_limit or 1000
        self.mini_batch_size = parameters.mini_batch_size or 0
        self.convergence_decimals = parameters.convergence_decimals or 2
        self.shuffle = parameters.shuffle or False
        self.seed_instance = parameters.seed_instance
        self.instance_prng = RandomState(seed=self.seed_instance)
        self.seed_model = parameters.seed_model
        self.model_prng = RandomState(seed=self.seed_model)
        self.combiner = LTFArray.combiner_xor
        self.transformation = LTFArray.transform_lightweight_secure
        self.seed_challenge = parameters.seed_challenge
        self.challenge_prng = RandomState(self.seed_challenge)
        self.seed_distance = parameters.seed_distance
        self.distance_prng = RandomState(self.seed_distance)
        self.instance = None
        self.learner = None
        self.model = None
        self.training_set = None
        self.validation_set = None

    def run(self):
        self.instance = LTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, random_instance=self.instance_prng),
            transform=self.transformation,
            combiner=self.combiner,
            bias=0.0
        )
        validation_size = int((self.N / 1.1) // 10)
        self.training_set = TrainingSet(instance=self.instance, N=self.N - validation_size,
                                        random_instance=self.challenge_prng)
        self.validation_set = TrainingSet(instance=self.instance, N=validation_size,
                                          random_instance=self.distance_prng)
        self.learner = CorrelationAttack(
            n=self.n,
            k=self.k,
            training_set=self.training_set,
            validation_set=self.validation_set,
            weights_prng=self.model_prng,
            lr_iteration_limit=self.lr_iteration_limit,
            mini_batch_size=self.mini_batch_size,
            convergence_decimals=self.convergence_decimals,
            shuffle=self.shuffle,
            logger=self.progress_logger,
        )
        self.model = self.learner.learn()

    def analyze(self):
        """
        Analyzes the learned result.
        """
        assert self.model is not None

        accuracy = 1.0 - approx_dist(
            self.instance,
            self.model,
            min(10000, 2 ** self.n),
            self.distance_prng
        )

        correct_iteration = None
        if self.learner.total_permutation_iterations > 0:
            correct_iteration = self.find_correct_permutation(self.learner.initial_model.weight_array)

        return Result(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            initial_model=self.learner.initial_model,
            initial_lr_iterations=self.learner.initial_lr_iterations,
            initial_accuracy=self.learner.initial_accuracy,
            correct_permutation=correct_iteration,
            best_permutation=self.learner.best_permutation,
            best_permutation_iteration=self.learner.best_permutation_iteration,
            total_permutation_iterations=self.learner.total_permutation_iterations,
            total_lr_iterations=self.learner.total_lr_iterations,
            model=self.model,
            accuracy=accuracy
        )

    def find_correct_permutation(self, weights):
        """
        Finds the best permutation of the given weights to fit the original instance weights.
        :param weights: The weight-array to permute
        :return: The best permutation
        """
        instance_weights = self.instance.weight_array

        max_correlation = 0
        best_permutation = None
        for permutation in list(permutations(range(self.k))):
            adopted_model_weights = self.learner.adopt_weights(weights, permutation)
            assert adopted_model_weights.shape == (self.k, self.n + 1), \
                'adopted weights shape is %s but was expected to be (%i, %i)' % (
                    str(adopted_model_weights.shape),
                    self.k,
                    self.n + 1
                )
            assert instance_weights.shape == (self.k, self.n + 1)
            correlation = [
                abs(pearsonr(
                    abs(adopted_model_weights[l] / norm(adopted_model_weights[l])),
                    abs(instance_weights[l] / norm(instance_weights[l]))
                )[0])
                for l in range(self.k)
            ]

            if sum(correlation) > max_correlation:
                max_correlation = sum(correlation)
                best_permutation = permutation

        return best_permutation
