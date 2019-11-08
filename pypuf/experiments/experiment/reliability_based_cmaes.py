""" This module provides an experiment class which learns an instance of LTFArray with
    reliability based CMAES learner.
    It is based on the work from G. T. Becker in "The Gap Between Promise and Reality: On
    the Insecurity of XOR Arbiter PUFs". The learning algorithm applies Covariance Matrix
    Adaptation Evolution Strategies from N. Hansen in "The CMA Evolution Strategy: A
    Comparing Review".
"""

from os import getpid
from typing import NamedTuple
from uuid import UUID

import numpy as np
from numpy.random.mtrand import RandomState
from scipy.stats import pearsonr

from pypuf.tools import approx_dist, TrainingSet
from pypuf.experiments.experiment.base import Experiment
# from pypuf.learner.evolution_strategies.reliability_based_cmaes import ReliabilityBasedCMAES
from pypuf.learner.evolution_strategies.reliability_cmaes_learner import ReliabilityBasedCMAES
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray


class Parameters(NamedTuple):
    n: int
    k: int
    seed_instance: int
    seed_model: int
    seed_challenges: int
    transform: str
    combiner: str
    noisiness: float
    num: int
    reps: int
    pop_size: int
    abort_delta: float
    abort_iter: int


class Result(NamedTuple):
    experiment_id: UUID
    measured_time: float
    pid: int
    accuracy: float
    iterations: int
    abortions: int
    stops: str
    max_possible_acc: float
    cross_model_correlation: list
    discard_count: dict



class ExperimentReliabilityBasedCMAES(Experiment):
    """
        This class implements an experiment for executing the reliability based CMAES
        learner for XOR LTF arrays.
        It provides all relevant parameters as well as an instance of an LTF array to
        learn.
        Furthermore, the learning results are being logged into csv files.
    """

    def __init__(self, progress_log_name, parameters: Parameters):
        """
            Initialize an Experiment using the Reliability based CMAES Learner for
            modeling LTF Arrays.
            :param progress_log_name:   Log name, Prefix of the name of the experiment log
                                        file
            :param parameters:          Parameters object for this experiment
        """

        super().__init__(
            '%s.0x%x_%i_%i_%i_%i_%i' % (
                progress_log_name,
                parameters.seed_instance,
                parameters.k,
                parameters.n,
                parameters.num,
                parameters.reps,
                parameters.pop_size),
            parameters)
        self.prng_i = RandomState(seed=self.parameters.seed_instance)
        self.prng_c = RandomState(seed=self.parameters.seed_challenges)
        self.training_set = None
        self.instance = None
        self.learner = None
        self.model = None

    def run(self):
        """
            Initialize the instance, the training set and the learner
            to then run the Reliability based CMAES with the given parameters.
        """

        # Instantiate the baseline Noisy LTF Array that shall be learned
        self.instance = NoisyLTFArray(
                            weight_array=LTFArray.normal_weights(
                                self.parameters.n,
                                self.parameters.k,
                                random_instance=self.prng_i),
                            transform=self.parameters.transform,
                            combiner=self.parameters.combiner,
                            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(
                                self.parameters.n,
                                1,
                                self.parameters.noisiness),
                            random_instance=self.prng_i)

        # Sample training data from the Noisy LTF Array
        self.training_set = TrainingSet(
                                self.instance,
                                self.parameters.num,
                                self.prng_c,
                                self.parameters.reps)

        # Instantiate the CMA-ES learner
        self.learner = ReliabilityBasedCMAES(
                           self.training_set,
                           self.parameters.k,
                           self.parameters.n,
                           self.instance.transform,
                           self.instance.combiner,
                           self.parameters.pop_size,
                           self.parameters.abort_delta,
                           self.parameters.abort_iter,
                           self.parameters.seed_model,
                           self.progress_logger)

        # Start learning a model
        self.model, self.discard_count = self.learner.learn()


    def analyze(self):
        """
            Analyze the results and return the Results object.
        """
        n = self.parameters.n

        # Accuracy of the learned model using 10000 random samples.
        empirical_accuracy     = 1 - approx_dist(self.instance, self.model,
                                    10000, RandomState(1902380))

        # Accuracy of the base line Noisy LTF. Can be < 1.0 since it is Noisy.
        best_empirical_accuracy = 1 - approx_dist(self.instance,
                                    LTFArray(
                                        weight_array=self.instance.weight_array[:, :n],
                                        transform=self.parameters.transform,
                                        combiner=self.parameters.combiner),
                                    10000, RandomState(12346))
        # Correl. of the learned model and the base line LTF using pearson for all chains
        cross_model_correlation = [[pearsonr(v[:n], w[:n])[0]
                                        for w in self.model.weight_array]
                                        for v in self.training_set.instance.weight_array]

        return Result(
            experiment_id=self.id,
            measured_time=self.measured_time,
            pid=getpid(),
            accuracy=empirical_accuracy,
            iterations=self.learner.num_iterations,
            abortions=self.learner.num_abortions,
            stops=self.learner.stops,
            max_possible_acc=best_empirical_accuracy,
            cross_model_correlation=cross_model_correlation,
            discard_count=self.discard_count
        )
