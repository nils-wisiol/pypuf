"""This module provides experiments which can be used to calculate the Fourier coefficients of a pypuf.simulation."""
from numpy.random import RandomState
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.pac.low_degree import LowDegreeAlgorithm
from pypuf.tools import TrainingSet


class ExperimentFCCRP(Experiment):
    """This Experiment calculates the Fourier coefficients for a pypuf.simulation instance."""

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
        results = '{}\t{}\t{}'.format(instance_parameter_str, fourier_coefficient_str, self.measured_time)
        self.result_logger.info(results)
