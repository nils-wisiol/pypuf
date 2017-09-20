"""This module tests the logistic regression learner."""
import unittest
from numpy.random import RandomState
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.tools import TrainingSet


class TestLogisticRegression(unittest.TestCase):
    """
    This module tests the logistic regression learner.
    """
    n = 8
    k = 2
    N = 2**8
    seed_model = 1234
    seed_instance = 1234

    def test_learn_xor(self):
        """"
        Stupid test which gains code coverage
        """
        instance_prng = RandomState(seed=TestLogisticRegression.seed_instance)
        model_prng = RandomState(seed=TestLogisticRegression.seed_model)

        instance = LTFArray(
            weight_array=LTFArray.normal_weights(
                TestLogisticRegression.n,
                TestLogisticRegression.k,
                random_instance=instance_prng
            ),
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
        )

        lr_learner = LogisticRegression(
            TrainingSet(instance=instance, N=TestLogisticRegression.N),
            TestLogisticRegression.n,
            TestLogisticRegression.k,
            transformation=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            weights_prng=model_prng,
        )
        lr_learner.learn()

    def test_learn_ip_mod2(self):
        """"
        Stupid test which gains code coverage
        """
        instance_prng = RandomState(seed=TestLogisticRegression.seed_instance)
        model_prng = RandomState(seed=TestLogisticRegression.seed_model)

        instance = LTFArray(
            weight_array=LTFArray.normal_weights(
                TestLogisticRegression.n,
                TestLogisticRegression.k,
                random_instance=instance_prng
            ),
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_ip_mod2,
        )

        lr_learner = LogisticRegression(
            TrainingSet(instance=instance, N=TestLogisticRegression.N),
            TestLogisticRegression.n,
            TestLogisticRegression.k,
            transformation=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            weights_prng=model_prng,
        )
        lr_learner.learn()
