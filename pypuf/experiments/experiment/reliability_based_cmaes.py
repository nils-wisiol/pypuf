"""This module provides an experiment class which learns an instance of LTFArray with reliability based CMAES learner.
It is based on the work from G. T. Becker in "The Gap Between Promise and Reality: On the Insecurity of
XOR Arbiter PUFs". The learning algorithm applies Covariance Matrix Adaptation Evolution Strategies from N. Hansen
in "The CMA Evolution Strategy: A Comparing Review".
"""
import numpy as np

from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.evolution_strategies.reliability_based_cmaes import ReliabilityBasedCMAES
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf import tools


class ExperimentReliabilityBasedCMAES(Experiment):
    """This class implements an experiment for executing the reliability based CMAES learner for XOR LTF arrays.
    It provides all relevant parameters as well as an instance of an LTF array to learn.
    Furthermore, the learning results are being logged into csv files.
    """

    def __init__(
            self, log_name,
            seed_instance, k, n, transform, combiner, noisiness,
            seed_challenges, num, reps,
            seed_model, pop_size, limit_stag, limit_iter
    ):
        """Initialize an Experiment using the Reliability based CMAES Learner for modeling LTF Arrays
        :param log_name:        Log name, Prefix of the name of the experiment log file
        :param seed_instance:   PRNG seed used to create an LTF array instance to learn
        :param k:               Width, the number of parallel LTFs in the LTF array
        :param n:               Length, the number stages within the LTF array
        :param transform:       Transformation function, the function that modifies the input within the LTF array
        :param combiner:        Combiner, the function that combines particular chains' outputs within the LTF array
        :param noisiness:       Noisiness, the relative scale of noise of instance compared to the scale of weights
        :param seed_challenges: PRNG seed used to sample challenges
        :param num:             Challenge number, the number of binary inputs (challenges) for the LTF array
        :param reps:            Repetitions, the number of evaluations of every challenge (part of training_set)
        :param seed_model:      PRNG seed used by the CMAES algorithm for sampling solution points
        :param pop_size:        Population size, the number of sampled points of every CMAES iteration
        :param limit_stag:      Stagnation limit, the maximal number of stagnating iterations within the CMAES
        :param limit_iter:      Iteration limit, the maximal number of iterations within the CMAES
        """
        super().__init__(
            log_name='%s.0x%x_%i_%i_%i_%i_%i' % (
                log_name,
                seed_instance,
                k,
                n,
                num,
                reps,
                pop_size,
            ),
        )
        # Instance of LTF array to learn
        self.seed_instance = seed_instance
        self.prng_i = np.random.RandomState(seed=self.seed_instance)
        self.k = k
        self.n = n
        self.transform = transform
        self.combiner = combiner
        self.noisiness = noisiness
        # Training set
        self.seed_challenges = seed_challenges
        self.prng_c = np.random.RandomState(seed=self.seed_instance)
        self.num = num
        self.reps = reps
        self.training_set = None
        # Parameters for CMAES
        self.seed_model = seed_model
        self.pop_size = pop_size
        self.limit_s = limit_stag
        self.limit_i = limit_iter
        # Basic objects
        self.instance = None
        self.learner = None
        self.model = None

    def run(self):
        """Initialize the instance, the training set and the learner
        to then run the Reliability based CMAES with the given parameters
        """
        self.instance = NoisyLTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, random_instance=self.prng_i),
            transform=self.transform,
            combiner=self.combiner,
            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(self.n, 1, self.noisiness),
            random_instance=self.prng_i,
        )
        self.training_set = tools.TrainingSet(self.instance, self.num, self.prng_c, self.reps)
        self.learner = ReliabilityBasedCMAES(
            self.training_set,
            self.k,
            self.n,
            self.transform,
            self.combiner,
            self.pop_size,
            self.limit_s,
            self.limit_i,
            self.seed_model,
            self.progress_logger,
        )
        self.model = self.learner.learn()

    def analyze(self):
        """Analyze the learned model"""
        assert self.model is not None
        self.result_logger.info(
            '0x%x\t0x%x\t0x%x\t%i\t%i\t%i\t%f\t%i\t%i\t%f\t%s\t%f\t%s\t%i\t%i\t%s',
            self.seed_instance,
            self.seed_challenges,
            self.seed_model,
            self.n,
            self.k,
            self.num,
            self.noisiness,
            self.reps,
            self.pop_size,
            1.0 - tools.approx_dist(self.instance, self.model, min(100000, 2 ** self.n), self.prng_c),
            ','.join(map(str, self.calc_individual_accs())),
            self.measured_time,
            self.learner.stops,
            self.learner.num_abortions,
            self.learner.num_iterations,
            ','.join(map(str, self.model.weight_array.flatten() / np.linalg.norm(self.model.weight_array.flatten()))),
        )

    def calc_individual_accs(self):
        """Calculate the accuracies of individual chains of the learned model"""
        transform = self.model.transform
        combiner = self.model.combiner
        accuracies = np.zeros(self.k)
        poles = np.zeros(self.k)
        for i in range(self.k):
            chain_original = LTFArray(self.instance.weight_array[i, np.newaxis, :], transform, combiner)
            for j in range(self.k):
                chain_model = LTFArray(self.model.weight_array[j, np.newaxis, :], transform, combiner)
                accuracy = tools.approx_dist(chain_original, chain_model, min(10000, 2 ** self.n), self.prng_c)
                pole = 1
                if accuracy < 0.5:
                    accuracy = 1.0 - accuracy
                    pole = -1
                if accuracy > accuracies[i]:
                    accuracies[i] = accuracy
                    poles[i] = pole
        accuracies *= poles
        for i in range(self.k):
            if accuracies[i] < 0:
                accuracies[i] += 1
        return accuracies
