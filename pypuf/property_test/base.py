"""
This module provides a class for several property tests which can be used to check the attributes of an PUF.
"""
from numpy import array, mean, median, sqrt, sign, reshape, broadcast_to
from numpy import min as np_min
from numpy import max as np_max
from numpy import sum as np_sum


class PropertyTest(object):
    """
    This class executes essential routines for each property test.
    The set of instances is expected to be homogeneous in n the number of stages.
    """

    def __init__(self, instances, logger=None):
        """
        :param instances: list of pypuf.simulation.base.Simulation
        """
        self.instances = instances
        self.logger = logger

    @staticmethod
    def statistic(float_set):
        """
        This function calculates standard statistical metrics over a list of floats.
        :param float_set: list of float
        :return: Dictionary of float and list of float
                 Statistic of the set of instances {mean:, median:, min:, max:, sample_variance:, samples:}
        """
        float_set_size = len(float_set)

        # Calculate the sample mean
        factor = 1 / float_set_size
        sample_mean = factor * np_sum(float_set)

        # Calculate the standard deviation
        factor = 1 / (float_set_size - 1)
        sample_variance = sqrt(factor * np_sum((float_set - sample_mean) ** 2))

        minimum = np_min(float_set)
        maximum = np_max(float_set)

        median_dist = median(float_set)

        return {
            'mean': sample_mean,
            'median': median_dist,
            'min': minimum,
            'max': maximum,
            'sv': sample_variance,
            'samples': float_set
        }

    @staticmethod
    def reliability(instance, challenge, measurements=10):
        """
        This function calculates the reliability of a puf instance.
        :param instance: pypuf.simulation.base.Simulation
        :param challenge: array of int shape(1,n)
        :param measurements: int default 10
                             Number of evaluations of the puf instance.
        :return: float
                 The reliability for Reliability in percent.
        """
        # Calculate the responses
        challenges = broadcast_to(challenge, (measurements,) + challenge.shape)
        responses = instance.eval(challenges)

        # Approximate the real response by majority vote over the measurements
        real_response = sign(np_sum(responses, axis=0))

        # If we get simulations with response length > 1 then change the calculation to use the hamming distance.
        # This matrix contains the distances of responses of a challenge for r evaluations compared with real_response
        response_distances = responses != real_response
        return mean(response_distances)

    @staticmethod
    def reliability_set(instances, challenges, measurements=10):
        """
        This function calculates a set of reliabilities.
        :param instances: instances: list of pypuf.simulation.base.Simulation
        :param challenges: array of int shape(N,n)
        :param measurements: int default 10
                             Number of evaluations of the puf instance.
        :return: list of float
                 Array of reliabilities for the puf instances, challenges and measurements.
        """
        reliabilities = []
        for ins in instances:
            for challenge in challenges:
                reliabilities.append(PropertyTest.reliability(ins, challenge, measurements=measurements))
        return reliabilities

    def reliability_statistic(self, challenges, measurements=10):
        """
        This function calculates the reliability statistic.
        :param challenges: array of int shape(N,n)
        :param measurements: int default 10
                             Number of evaluations of the puf instance.
        :return: Dictionary of float and list of float
                 Statistic of the set of instances {mean:, median:, min:, max:, sample_variance:, samples:}
        """
        return PropertyTest.statistic(
            PropertyTest.reliability_set(self.instances, challenges, measurements=measurements)
        )

    @staticmethod
    def uniqueness(instances, challenge):
        """
        This function calculates the uniqueness of a challenge response for a set of simulation instances.
        :param instances: array of pypuf.simulation.base.Simulation with shape(k)
        :param challenge: challenge: array of int shape(1,n)
        :return: float
                 Uniqueness in percent.
        """
        # If we get simulations with response length > 1 then change response extraction.
        responses = array([instance.eval(array([challenge]))[0] for instance in instances])
        m = len(instances)
        distance_sum = 0
        for u in range(m - 1):
            for v in range(u + 1, m):
                # If we get simulations with response length > 1 then change the calculation of the distance.
                distance_sum = distance_sum + (responses[u] != responses[v])
        # Arithmetic mean of the sum of response distances
        return 2 / (m * (m - 1)) * distance_sum

    @staticmethod
    def uniqueness_set(instances, challenges, measurements=1):
        """
        This function calculates a uniqueness set for a list of instances, challenges and measurements.
        :param instances: array of pypuf.simulation.base.Simulation with shape(k)
        :param challenges: challenge: array of int shape(N,n)
        :param measurements: int default 10
                             Number of uniqueness calculation for fix arguments.
        :return: list of float
                 List of uniqueness.
        """
        uniqueness_set = []
        for challenge in challenges:
            for _ in range(measurements):
                uniqueness_set.append(PropertyTest.uniqueness(instances, challenge))
        return uniqueness_set

    def uniqueness_statistic(self, challenges, measurements=10):
        """
        This function generates a statistic about the uniqueness of a set of simulation instances a set of challenges
        and repeated measurements.
        :param challenges: array of int shape(N,n)
        :param measurements: int default 10
                             Number of uniqueness calculation for fix arguments.
        :return: Dictionary of float and list of float
                 Statistic of the set of instances {mean:, median:, min:, max:, sample_variance:, samples:}
        """
        return PropertyTest.statistic(
            PropertyTest.uniqueness_set(self.instances, challenges, measurements=measurements)
        )
