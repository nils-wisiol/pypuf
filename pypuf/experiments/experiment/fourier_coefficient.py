"""This module provides experiments which can be used to estimate the Fourier coefficients of a pypuf.simulation."""
from numpy.random import RandomState
from numpy import matmul, zeros, cumsum, array, float64, tile
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.pac.low_degree import LowDegreeAlgorithm
from pypuf.simulation.fourier_based.dictator import Dictator
from pypuf.simulation.fourier_based.bent import BentFunctionIpMod2
from pypuf.tools import TrainingSet, sample_inputs, random_inputs, RESULT_TYPE


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
    def create_bent_instances(cls, instance_count=1, n=8):
        """
        This function creates a bent function simulation.
        :param instance_count: int
                               Number of instances.
        :param n: int
                  Number of input bits.
        :return: list of BentFunctionIpMod2
        """
        return [BentFunctionIpMod2(n) for _ in range(instance_count)]


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
        """
        This function approximates a degree one weight based on the following parameters.
        :param responses: array of float or pypuf.tools.RESULT_TYPE shape(N)
                          Array of responses for the N different challenges.
        :param challenges:  array of int8 shape(N,n)
                            Array of challenges related the responses.
        :param challenge_count_min: int
        :param challenge_count_max: int
        :return: list of float
                 Degree one weights.
        """
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
        results = '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            instance_parameter_str,
            self.challenge_seed,
            self.challenge_count_min,
            self.challenge_count_max,
            degree_one_str,
            self.measured_time,
            unique_id
        )
        self.result_logger.info(results)


class ExperimentCFCAMatules(Experiment):
    """
    This class can be used to approximate degree-one weights through an cumulative sum, using the method from
    K Matulef, R O’Donnell, R Rubinfeld, and R Servedio. Testing halfspaces.
    SIAM J. Comput., 39(5):2004–2047, January 2010.
    """

    def __init__(
            self, log_name, challenge_count_min, challenge_count_max, challenge_seed, mu, instance_gen,
            instance_parameter
    ):
        """
        :param log_name: string
                         Name of the progress log.
        :param challenge_count_min: int
                                Minimum number of challenges which are used to approximate the Fourier coefficients.
                                The number will be multiplied with 3 regarding Matulef et al. approximation.
        :param challenge_count_max: int
                                Maximum number of challenges which are used to approximate the Fourier coefficients.
                                The number will be multiplied with 3 regarding Matulef et al. approximation.
        :param challenge_seed: int
                               Seed which is used to generate uniform at random distributed challenges.
        :param mu: float
                   Desired additive absolute approximation error.
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
        self.mu = float64(mu)
        self.instance_gen = instance_gen
        self.instance_parameter = instance_parameter
        self.results = []

    @classmethod
    def approx_degree_one_weight(
            cls,
            responses_x1,
            responses_x2,
            responses_x1_y,
            challenge_count_min,
            challenge_count_max,
            mu
    ):
        """
        This function approximates a degree one weight based on the following parameters.
        :param responses_x1: array of float or pypuf.tools.RESULT_TYPE shape(N)
                          Array of responses for the N different challenges_x1.
        :param responses_x2: array of float or pypuf.tools.RESULT_TYPE shape(N)
                          Array of responses for the N different challenges_x2.
        :param responses_x1_y: array of float or pypuf.tools.RESULT_TYPE shape(N)
                          Array of responses for the N different challenges_x1 * y.
        :param challenge_count_min: int
        :param challenge_count_max: int
        :return: array of float
                 Degree one weights.
        """
        responses_x1_x2 = responses_x1 * responses_x2
        empty_set_weight_sum = cumsum(responses_x1_x2, dtype='float64')
        weight_sum = cumsum(responses_x1 * responses_x1_y, dtype='float64')
        mean_divisor = array(list(range(challenge_count_min, challenge_count_max + 1)), dtype='float64')
        empty_set_weight = empty_set_weight_sum[challenge_count_min - 1:] / mean_divisor
        weights = weight_sum[challenge_count_min - 1:] / mean_divisor
        mus = mu ** 2
        results = (weights - empty_set_weight - mus) / mu
        return results

    def run(self):
        """This method executes the Fourier coefficient approximation"""
        challenge_prng = RandomState(self.challenge_seed)
        instance = self.instance_gen(**self.instance_parameter)[0]
        # Calculate all challenge response pairs
        challenges_x1 = random_inputs(instance.n, self.challenge_count_max, random_instance=challenge_prng)
        challenges_x2 = random_inputs(instance.n, self.challenge_count_max, random_instance=challenge_prng)
        prob_plus_one = 1/2 + ((1/2) * self.mu)
        prob_minus_one = 1 - prob_plus_one
        y = (challenge_prng.choice(
            [-1, +1],
            size=(self.challenge_count_max, instance.n),
            p=[prob_minus_one, prob_plus_one]
        )).astype(RESULT_TYPE)
        combined_challenges = challenges_x1 * y
        responses_x1 = instance.eval(challenges_x1)
        responses_x2 = instance.eval(challenges_x2)
        responses_x1_y = instance.eval(combined_challenges)


        self.results = ExperimentCFCAMatules.approx_degree_one_weight(
            responses_x1, responses_x2, responses_x1_y, self.challenge_count_min, self.challenge_count_max, self.mu
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
        results = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            instance_parameter_str,
            self.challenge_seed,
            self.challenge_count_min,
            self.challenge_count_max,
            self.mu,
            degree_one_str,
            self.measured_time,
            unique_id
        )
        self.result_logger.info(results)



class ExperimentArbiterPUFCFCA(Experiment):
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
        """
        This function approximates a degree one weight based on the following parameters.
        :param responses: array of float or pypuf.tools.RESULT_TYPE shape(N)
                          Array of responses for the N different challenges.
        :param challenges:  array of int8 shape(N,n)
                            Array of challenges related the responses.
        :param challenge_count_min: int
        :param challenge_count_max: int
        :return: list of float
                 Degree one weights.
        """
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

    @classmethod
    def inverse_atf(cls, challenges):
        """This functions can be used to inverse the atf transformation."""
        n = len(challenges)
        result = zeros(n, dtype='int8')
        count_minus_one = 0
        for i in range(n - 1, -1, -1):
            if challenges[i] == 1 and count_minus_one % 2 == 0:
                result[i] = 1
                continue
            if challenges[i] == 1 and count_minus_one % 2 == 1:
                result[i] = -1
                count_minus_one += 1
                continue
            if challenges[i] == -1 and count_minus_one % 2 == 0:
                result[i] = -1
                count_minus_one += 1
                continue
            if challenges[i] == -1 and count_minus_one % 2 == 1:
                result[i] = 1

        return result

    def run(self):
        """This method executes the Fourier coefficient approximation"""
        challenge_prng = RandomState(self.challenge_seed)
        instance = self.instance_gen(**self.instance_parameter)[0]
        # Calculate all challenge response pairs
        challenges = sample_inputs(instance.n, self.challenge_count_max, random_instance=challenge_prng)
        challenge_prng.shuffle(challenges)
        for i in range(self.challenge_count_max):
            challenges[i] = ExperimentArbiterPUFCFCA.inverse_atf(challenges[i])
        responses = instance.eval(challenges)
        self.results = ExperimentArbiterPUFCFCA.approx_degree_one_weight(
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
        results = '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            instance_parameter_str,
            self.challenge_seed,
            self.challenge_count_min,
            self.challenge_count_max,
            degree_one_str,
            self.measured_time,
            unique_id
        )
        self.result_logger.info(results)
