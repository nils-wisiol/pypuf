"""This module provides experiments which can be used to estimate the Fourier coefficients of a pypuf.simulation."""
import time
from numpy.random import RandomState, shuffle
from numpy import matmul
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.pac.low_degree import LowDegreeAlgorithm
from pypuf.simulation.fourier_based.dictator import Dictator
from pypuf.simulation.fourier_based.bent import BentFunction
from pypuf.tools import TrainingSet, sample_inputs


class ExperimentFCCRP(Experiment):
    """This Experiment estimates the Fourier coefficients for a pypuf.simulation instance."""

    def __init__(self, log_name, challenge_count, challenge_seed, instance_gen, instance_parameter):
        """
        :param log_name: string
                         Name of the progress log.
        :param challenge_count: int
                                Number of challenges which are used to approximate the Fourier coefficients.
        :param challenge_seed: int
                               Seed which is used to generate uniform at random distributed challenges.
        :param instance_gen: function
                             Function which is used to create an instance to approximate Fourier coefficients.
        :param instance_parameter: A collections.OrderedDict with keyword arguments
                                   This keyword arguments are passed to instance_gen to generate a
                                   pypuf.simulation.base.Simulation instances.
        """
        self.log_name = log_name
        super().__init__(self.log_name)
        self.challenge_count = challenge_count
        self.challenge_seed = challenge_seed
        self.instance_gen = instance_gen
        self.instance_parameter = instance_parameter
        self.fourier_coefficients = []

    def run(self):
        """This method executes the degree-1 Fourier coefficient calculation."""
        challenge_prng = RandomState(self.challenge_seed)
        instance = self.instance_gen(**self.instance_parameter)[0]
        training_set = TrainingSet(instance, self.challenge_count, random_instance=challenge_prng)
        degree_1_learner = LowDegreeAlgorithm(training_set=training_set, degree=1)
        self.fourier_coefficients = (degree_1_learner.learn()).fourier_coefficients
        self.fourier_coefficients = [str(coefficient.val) for coefficient in self.fourier_coefficients]

    def analyze(self):
        """This method logs the Fourier coefficient experiment result."""
        instance_param = []
        for value in self.instance_parameter.values():
            if callable(value):
                instance_param.append(value.__name__)
            else:
                instance_param.append(str(value))
        instance_parameter_str = '\t'.join(instance_param)
        fourier_coefficient_str = ','.join(self.fourier_coefficients)
        unique_id = '{}{}{}'.format(
            ''.join(instance_param), self.challenge_count, self.challenge_seed
        )
        results = '{}\t{}\t{}\t{}\t{}\t{}'.format(
            instance_parameter_str,
            self.challenge_seed,
            self.challenge_count,
            fourier_coefficient_str,
            self.measured_time,
            unique_id
        )
        self.result_logger.info(results)

    @classmethod
    def create_dictator_instances(cls, instance_count=1, n=8, dictator=0):
        """
        This function can be used to create a list of dictator simulations.
        :param instance_count: int
                               Number of dictator simulations to create.
        :param n: int
                  Number of input bits
        :param dictator: int
                         Index for dictatorship
        :return: list of pypuf.simulation.fourier_based.dictator.Dictator
        """
        return [Dictator(dictator, n) for _ in range(instance_count)]

    @classmethod
    def create_bent_instances(cls, instance_count=1, n=8, name='ipmod_2'):
        return  [BentFunction(n) for _ in range(instance_count)]


class ExperimentCFCA(Experiment):
    """
    This class can be used to approximate Fourier coefficients through an cumulative sum, which is faster than multiple
    ExperimentFCCRP experiments.
    """

    def __init__(
            self, log_name, challenge_count_min, challenge_count_max, challenge_seed, instance_gen, instance_parameter
    ):
        """
        :param log_name: string
                         Name of the progress log.
        :param challenge_count_min: int
                                Minimum number of challenges which are used to approximate the Fourier coefficients.
        :param challenge_count_max: int
                                Maximum number of challenges which are used to approximate the Fourier coefficients.
        :param challenge_seed: int
                               Seed which is used to generate uniform at random distributed challenges.
        :param instance_gen: function
                             Function which is used to create an instance to approximate Fourier coefficients.
        :param instance_parameter: A collections.OrderedDict with keyword arguments
                                   This keyword arguments are passed to instance_gen to generate a
                                   pypuf.simulation.base.Simulation instances.
        """
        assert challenge_count_min < challenge_count_max
        self.log_name = log_name
        super().__init__(self.log_name)
        self.challenge_count_min = challenge_count_min
        self.challenge_count_max = challenge_count_max
        self.challenge_seed = challenge_seed
        self.instance_gen = instance_gen
        self.instance_parameter = instance_parameter
        self.fourier_coefficients = []
        self.results = ''

        instance_param = []
        for value in self.instance_parameter.values():
            if callable(value):
                instance_param.append(value.__name__)
            else:
                instance_param.append(str(value))
        self.instance_parameter_str = '\t'.join(instance_param)
        self.clinched_instance_parameter_str = ''.join(instance_param)

    def run(self):
        """This method executes the Fourier coefficient approximation"""
        challenge_prng = RandomState(self.challenge_seed)
        instance = self.instance_gen(**self.instance_parameter)[0]
        # Calculate all challenge response pairs
        challenges = sample_inputs(instance.n, self.challenge_count_max, random_instance=challenge_prng)
        challenge_prng.shuffle(challenges)
        responses = instance.eval(challenges)
        start_time = time.time()
        # Calculate the Fourier coefficients for self.challenge_count_min challenges
        coefficient_sums = matmul(responses[:self.challenge_count_min], challenges[:self.challenge_count_min])
        coefficient_sums = coefficient_sums.astype('int64')
        self.fourier_coefficients = coefficient_sums / self.challenge_count_min
        self.log_fourier_coefficient(self.challenge_count_min, time.time() - start_time)

        # Calculate Fourier coefficients based on the previous response sum
        for i in range(self.challenge_count_min + 1, self.challenge_count_max):
            start_time = time.time()
            part_sum = responses[i] * challenges[i]
            coefficient_sums = coefficient_sums + part_sum
            self.fourier_coefficients = coefficient_sums / i
            measured_time = time.time() - start_time
            self.log_fourier_coefficient(i, measured_time)

    def analyze(self):
        """This method prints the results to the result logger"""
        self.result_logger.info(self.results)

    def log_fourier_coefficient(self, challenge_count, measured_time):
        """
        This method safes the experiment progress to the progress log and appends the partial result to self.result.
        """
        fourier_coefficient_str = ','.join(list(map(str, self.fourier_coefficients)))
        unique_id = '{}{}{}'.format(
            self.clinched_instance_parameter_str, challenge_count, self.challenge_seed
        )
        results = '{}\t{}\t{}\t{}\t{}\t{}'.format(
            self.instance_parameter_str,
            self.challenge_seed,
            challenge_count,
            fourier_coefficient_str,
            measured_time,
            unique_id
        )
        self.results += results + '\n'
        self.progress_logger.info(results)
