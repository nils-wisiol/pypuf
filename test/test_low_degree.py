"""
This module tests the low degree learner.
"""
import unittest
from numpy.random import RandomState
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.pac.low_degree import LowDegreeAlgorithm
from pypuf.tools import TrainingSet


class TestLowDegree(unittest.TestCase):
    """
    This module tests the low degree learner.
    """
    n = 8
    k = 2
    N = 2**8
    degree = 2
    seed_instance = 1234

    def test_learn_xor(self):
        """"
        Stupid test which gains code coverage
        """
        instance_prng = RandomState(seed=TestLowDegree.seed_instance)

        instance = LTFArray(
            weight_array=LTFArray.normal_weights(
                TestLowDegree.n,
                TestLowDegree.k,
                random_instance=instance_prng
            ),
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
        )

        low_degree_learner = LowDegreeAlgorithm(
            TrainingSet(instance=instance, N=TestLowDegree.N),
            degree=TestLowDegree.degree
        )
        low_degree_learner.learn()
