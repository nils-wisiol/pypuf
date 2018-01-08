"""This module is used to store some examples for the documentation"""
from numpy import array, reshape
from pypuf.simulation.arbiter_based.ltfarray import NoisyLTFArray
from pypuf.property_test.base import PropertyTest
from pypuf.tools import sample_inputs


def main():
    """This method is used to execute all example functions."""
    example_reliability()
    example_reliability_statistic()


def example_reliability():
    """This method shows how to use the PropertyTest.reliability function."""
    n = 8
    k = 8
    transformation = NoisyLTFArray.transform_id
    combiner = NoisyLTFArray.combiner_xor
    weights = NoisyLTFArray.normal_weights(n=n, k=k)
    instance = NoisyLTFArray(
        weight_array=weights,
        transform=transformation,
        combiner=combiner,
        sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(n, 0.5)
    )
    challenge = array([-1, 1, 1, 1, -1, 1, 1, 1])
    reliability = PropertyTest.reliability(instance, reshape(challenge, (1, n)))
    print('The reliability is {}.'.format(reliability))


def example_reliability_statistic():
    """This method shows hot to use the PropertyTest.reliability_statistic."""
    n = 8
    k = 1
    N = 2 ** n
    instance_count = 3
    measurements = 100
    transformation = NoisyLTFArray.transform_id
    combiner = NoisyLTFArray.combiner_xor
    weights = NoisyLTFArray.normal_weights(n=n, k=k)
    instances = [
        NoisyLTFArray(
            weight_array=weights,
            transform=transformation,
            combiner=combiner,
            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(n, 0.5)
        ) for _ in range(instance_count)
    ]
    challenges = array(list(sample_inputs(n, N)))
    property_test = PropertyTest(instances)
    reliability_statistic = property_test.reliability_statistic(challenges, measurements=measurements)
    print('The reliability statistic is {}.'.format(reliability_statistic))


def example_uniqueness():
    """
    This method shows the function which can be used to calculate the uniqueness of a set of simulation instances.
    """
    n = 8
    k = 1
    instance_count = 3
    transformation = NoisyLTFArray.transform_id
    combiner = NoisyLTFArray.combiner_xor
    weights = NoisyLTFArray.normal_weights(n=n, k=k)
    instances = [
        NoisyLTFArray(
            weight_array=weights,
            transform=transformation,
            combiner=combiner,
            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(n, weights)
        ) for _ in range(instance_count)
    ]
    challenge = array([-1, 1, 1, 1, -1, 1, 1, 1])
    uniqueness = PropertyTest.uniqueness(instances, reshape(challenge, (1, n)))
    print('The uniqueness is {}.'.format(uniqueness))


def example_uniqueness_statistic():
    """This method shows the uniqueness statistic function."""
    n = 8
    k = 1
    N = 2 ** n
    instance_count = 11
    measurements = 1
    transformation = NoisyLTFArray.transform_id
    combiner = NoisyLTFArray.combiner_xor
    weights = NoisyLTFArray.normal_weights(n=n, k=k)
    instances = [
        NoisyLTFArray(
            weight_array=weights,
            transform=transformation,
            combiner=combiner,
            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(n, weights)
        ) for _ in range(instance_count)
    ]

    challenges = array(list(sample_inputs(n, N)))
    property_test = PropertyTest(instances)
    uniqueness_statistic = property_test.uniqueness_statistic(challenges, measurements=measurements)
    print('The uniqueness statistic is {}.'.format(uniqueness_statistic))

if __name__ == '__main__':
    main()
