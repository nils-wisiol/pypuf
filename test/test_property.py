"""This module tests the different functions which can be used to determine simulation properties."""
import unittest
from numpy import array, mean, reshape, repeat
from numpy.testing import assert_array_equal
from numpy.random import RandomState
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.tools import sample_inputs
from pypuf.property_test.base import PropertyTest


class TestPropertyTest(unittest.TestCase):
    """This class tests the property testing class."""

    def test_reliability(self):
        """This method tests the test_reliability calculation."""
        n = 8
        k = 8
        N = 2 ** n
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        instance = LTFArray(
            weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1)),
            transform=transformation,
            combiner=combiner,
        )
        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB1A))))
        reliabilities = []
        for challenge in challenges:
            reliabilities.append(PropertyTest.reliability(instance, reshape(challenge, (1, n))))

        # For noiseless simulations the responses are always the same hence the reliability is 0%
        assert_array_equal(reliabilities, repeat(0.0, N))

        noisy_instance = NoisyLTFArray(
            weight_array=NoisyLTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1)),
            transform=transformation,
            combiner=combiner,
            sigma_noise=0.5,
            random_instance=RandomState(0x5015E),
        )
        for challenge in challenges:
            reliability = PropertyTest.reliability(noisy_instance, challenge)
            # For noisy simulations the responses should vary
            self.assertNotEqual(reliability, 0.0)

    def test_reliability_set(self):
        """This method tests the reliability_statistic calculation."""
        n = 8
        k = 3
        N = 2 ** n
        measurements = 10
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        instances = []
        instance_count = 3
        for i in range(instance_count):
            instance = LTFArray(
                weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1+i)),
                transform=transformation,
                combiner=combiner,
            )
            instances.append(instance)

        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB0))))

        reliability_set = PropertyTest.reliability_set(instances, challenges, measurements=measurements)
        # The result is an array like with N * k entries.
        self.assertEqual(len(reliability_set), N * instance_count)
        # For noiseless simulations the all reliabilities must be 0%
        assert_array_equal(reliability_set, repeat(0.0, N * instance_count))

        noisy_instances = []
        for i in range(instance_count):
            noisy_instance = NoisyLTFArray(
                weight_array=NoisyLTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1+i)),
                transform=transformation,
                combiner=combiner,
                sigma_noise=0.5,
                random_instance=RandomState(0x5015C+i),
            )
            noisy_instances.append(noisy_instance)

        noisy_reliability_set = PropertyTest.reliability_set(noisy_instances, challenges, measurements=measurements)
        # For a noisy simulation the mean reliability must differ from zero
        self.assertNotEqual(mean(noisy_reliability_set), 0.0)

    def test_reliability_statistic(self):
        """This method tests the reliability statistic of an instance set."""
        n = 8
        k = 1
        N = 2 ** n
        instance_count = 2
        measurements = 100
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor

        instances = [
            LTFArray(weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                     transform=transformation, combiner=combiner) for i in range(instance_count)
        ]

        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB10))))

        property_test = PropertyTest(instances)
        reliability_statistic = property_test.reliability_statistic(challenges, measurements=measurements)
        # For an noiseless set of simulations the reliability must be 0%
        for key, value in reliability_statistic.items():
            if key == 'sv':
                self.assertEqual(value, 0.0, '{}'.format(key))
            elif key == 'samples':
                self.assertEqual(len(value), instance_count * N, '{}'.format(key))
            else:
                self.assertEqual(value, 0.0, '{}'.format(key))

        noisy_instances = [
            NoisyLTFArray(
                weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                transform=transformation,
                combiner=combiner,
                sigma_noise=0.5,
                random_instance=RandomState(0xCABE),
            ) for i in range(instance_count)
        ]

        noisy_property_test = PropertyTest(noisy_instances)
        noisy_reliability_statistic = noisy_property_test.reliability_statistic(challenges, measurements=measurements)
        self.assertNotEqual(noisy_reliability_statistic['mean'], 0.0)

    def test_uniqueness(self):
        """
        This method tests the function which can be used to calculate the uniqueness of a set of simulation instances.
        """
        n = 8
        k = 1
        N = 2 ** n
        instance_count = 50
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        instances = [
            LTFArray(weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                     transform=transformation, combiner=combiner) for i in range(instance_count)
        ]
        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB10))))
        uniqueness = []
        for challenge in challenges:
            uniqueness.append(PropertyTest.uniqueness(instances, reshape(challenge, (1, n))))
        # For normal distributed weights is the expected uniqueness near 0.5
        self.assertEqual(round(mean(uniqueness), 1), 0.5)

    def test_uniqueness_set(self):
        """This method tests the uniqueness set generation function."""
        n = 8
        k = 1
        N = 2 ** n
        instance_count = 25
        measurements = 2
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        instances = [
            LTFArray(weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                     transform=transformation, combiner=combiner) for i in range(instance_count)
        ]
        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB10))))

        uniqueness_set = PropertyTest.uniqueness_set(instances, challenges, measurements=measurements)

        # Check uniqueness_set to have the expected number of elements
        self.assertEqual(len(uniqueness_set), N * measurements)
        # For normal distributed weights is the expected uniqueness near 0.5
        self.assertEqual(round(mean(uniqueness_set), 1), 0.5)

    def test_uniqueness_statistic(self):
        """This method tests the uniqueness statistic function."""
        n = 8
        k = 1
        N = 2 ** n
        instance_count = 11
        measurements = 1
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor

        instances = [
            LTFArray(weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                     transform=transformation, combiner=combiner) for i in range(instance_count)
        ]

        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB10))))

        property_test = PropertyTest(instances)
        uniqueness_statistic = property_test.uniqueness_statistic(challenges, measurements=measurements)
        # For normal distributed weights is the expected uniqueness near 0.5
        self.assertEqual(round(uniqueness_statistic['mean'], 1), 0.5)
