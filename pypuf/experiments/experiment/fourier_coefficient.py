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
        self.results = []

    @classmethod
    def approx_degree_one_weight(cls, responses, challenges, challenge_count_min, challenge_count_max):
        results = []
        # Calculate the Fourier coefficients for self.challenge_count_min challenges
        coefficient_sums = matmul(responses[:challenge_count_min], challenges[:challenge_count_min])
        coefficient_sums = coefficient_sums.astype('int64')
        fourier_coefficient = coefficient_sums / challenge_count_min
        degree_one_weight = matmul(fourier_coefficient, fourier_coefficient)
        results.append(degree_one_weight)

        # Calculate Fourier coefficients based on the previous response sum
        for i in range(challenge_count_min, challenge_count_max):
            partial_sum = (responses[i] * challenges[i])
            coefficient_sums = coefficient_sums + partial_sum
            degree_one_weight = matmul(coefficient_sums / i, coefficient_sums / i)
            results.append(degree_one_weight)

        return results

    def run(self):
        """This method executes the Fourier coefficient approximation"""
        challenge_prng = RandomState(self.challenge_seed)
        instance = self.instance_gen(**self.instance_parameter)[0]
        # Calculate all challenge response pairs
        challenges = sample_inputs(instance.n, self.challenge_count_max, random_instance=challenge_prng)
        challenge_prng.shuffle(challenges)
        responses = instance.eval(challenges)

        self.results = ExperimentCFCA.approx_degree_one_weight(
            responses, challenges, self.challenge_count_min, self.challenge_count_max
        )



    def analyze(self):
        """This method prints the results to the result logger"""
        instance_param = []
        for value in self.instance_parameter.values():

            if callable(value):
                instance_param.append(value.__name__)
            else:
                instance_param.append(str(value))

        unique_id = '{}{}{}{}'.format(
            ''.join(instance_param), self.challenge_count_min, self.challenge_count_max, self.challenge_seed
        )
        degree_one_str = ','.join(list(map(str, self.results)))
        instance_parameter_str = '\t'.join(instance_param)
        results = '{}\t{}\t{}\t{}\t{}\t{}'.format(
            instance_parameter_str,
            self.challenge_seed,
            self.challenge_count_min,
            self.challenge_count_max,
            degree_one_str,
            self.measured_time,
            unique_id
        )
        self.result_logger.info(results)
