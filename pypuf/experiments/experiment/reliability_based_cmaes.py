"""This module provides an experiment class which learns an instance of LTFArray with reliability based CMAES learner.
It is based on the work from G. T. Becker in "The Gap Between Promise and Reality: On the Insecurity of
XOR Arbiter PUFs". The learning algorithm applies Covariance Matrix Adaptation Evolution Strategies from N. Hansen
in "The CMA Evolution Strategy: A Comparing Review".
"""
from os import getpid
from typing import NamedTuple
from uuid import UUID

import numpy as np
from numpy.random.mtrand import RandomState

from pypuf import tools
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.evolution_strategies.reliability_based_cmaes import ReliabilityBasedCMAES
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
    limit_stag: int
    limit_iter: int


class Result(NamedTuple):
    experiment_id: UUID
    measured_time: float
    pid: int
    accuracy: float
    

class ExperimentReliabilityBasedCMAES(Experiment):
    """This class implements an experiment for executing the reliability based CMAES learner for XOR LTF arrays.
    It provides all relevant parameters as well as an instance of an LTF array to learn.
    Furthermore, the learning results are being logged into csv files.
    """

    def __init__(self, progress_log_name, parameters: Parameters):
        """Initialize an Experiment using the Reliability based CMAES Learner for modeling LTF Arrays
        :param progress_log_name:        Log name, Prefix of the name of the experiment log file
        :param parameters Parameters object for this experiment
        """
        super().__init__(
            '%s.0x%x_%i_%i_%i_%i_%i' % (
                progress_log_name,
                parameters.seed_instance,
                parameters.k,
                parameters.n,
                parameters.num,
                parameters.reps,
                parameters.pop_size,
            ),
            parameters,
        )
        self.prng_i = RandomState(seed=self.parameters.seed_instance)
        self.prng_c = RandomState(seed=self.parameters.seed_challenges)
        self.training_set = None
        self.instance = None
        self.learner = None
        self.model = None

    def run(self):
        """Initialize the instance, the training set and the learner
        to then run the Reliability based CMAES with the given parameters
        """
        self.instance = NoisyLTFArray(
            weight_array=LTFArray.normal_weights(self.parameters.n, self.parameters.k, random_instance=self.prng_i),
            transform=self.parameters.transform,
            combiner=self.parameters.combiner,
            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(self.parameters.n, 1, self.parameters.noisiness),
            random_instance=self.prng_i,
        )
        self.training_set = tools.TrainingSet(self.instance, self.parameters.num, self.prng_c, self.parameters.reps)
        self.learner = ReliabilityBasedCMAES(
            self.training_set,
            self.parameters.k,
            self.parameters.n,
            self.parameters.transform,
            self.parameters.combiner,
            self.parameters.pop_size,
            self.parameters.limit_stag,
            self.parameters.limit_iter,
            self.parameters.seed_model,
            self.progress_logger,
        )
        # import sys
        # from time import time, clock
        # timer = clock if sys.platform == 'win32' else time
        # start_time = timer()
        self.model = self.learner.learn()
        # measured_time = timer() - start_time
        # print("measured time: " + str(measured_time))
        # print("distance")
        # print(1.0 - tools.approx_dist(self.instance, self.model, min(100000, 2 ** self.parameters.n), self.prng_c))
        # print(','.join(map(str, self.calc_individual_accs())))
        # print("seed")
        # print(self.parameters.seed_instance)
        # print("learner stops")
        # print(self.learner.stops)

    def analyze(self):
        """Analyze the learned model"""
        assert self.model is not None
        self.result_logger.info(
            '0x%x\t0x%x\t0x%x\t%i\t%i\t%i\t%f\t%i\t%i\t%f\t%s\t%f\t%s\t%i\t%i\t%s',
            self.parameters.seed_instance,
            self.parameters.seed_challenges,
            self.parameters.seed_model,
            self.parameters.n,
            self.parameters.k,
            self.parameters.num,
            self.parameters.noisiness,
            self.parameters.reps,
            self.parameters.pop_size,
            1.0 - tools.approx_dist(self.instance, self.model, min(10000, 2 ** self.parameters.n), self.prng_c),
            ','.join(map(str, self.calc_individual_accs())),
            self.measured_time,
            self.learner.stops,
            self.learner.num_abortions,
            self.learner.num_iterations,
            ','.join(map(str, self.model.weight_array.flatten() / np.linalg.norm(self.model.weight_array.flatten()))),
        )
        return Result(
            experiment_id=self.id,
            measured_time=self.measured_time,
            pid=getpid(),
            accuracy=1.0 - tools.approx_dist(self.instance, self.model, min(10000, 2 ** self.parameters.n), self.prng_c),
        )


    def calc_individual_accs(self):
        """Calculate the accuracies of individual chains of the learned model"""
        transform = self.model.transform
        combiner = self.model.combiner
        accuracies = np.zeros(self.parameters.k)
        poles = np.zeros(self.parameters.k)
        for i in range(self.parameters.k):
            chain_original = LTFArray(self.instance.weight_array[i, np.newaxis, :], transform, combiner)
            for j in range(self.parameters.k):
                chain_model = LTFArray(self.model.weight_array[j, np.newaxis, :], transform, combiner)
                accuracy = tools.approx_dist(chain_original, chain_model, min(10000, 2 ** self.parameters.n), self.prng_c)
                pole = 1
                if accuracy < 0.5:
                    accuracy = 1.0 - accuracy
                    pole = -1
                if accuracy > accuracies[i]:
                    accuracies[i] = accuracy
                    poles[i] = pole
        accuracies *= poles
        for i in range(self.parameters.k):
            if accuracies[i] < 0:
                accuracies[i] += 1
        return accuracies
