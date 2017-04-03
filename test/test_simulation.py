import unittest
from pypuf import simulation
from numpy.testing import assert_array_equal
from numpy import shape, dot, random, prod


class TestCombiner(unittest.TestCase):

    def test_combine_xor(self):
        assert_array_equal(
            simulation.LTFArray.combiner_xor(
                [
                    [ 1,  1, -3,  1],
                    [-1, -1, -1,  1],
                    [-2, -2,  2,  1]
                ]
            ),
            [
                -3,
                -1,
                8
            ]
        )


class TestInputTransformation(unittest.TestCase):

    def test_id(self):
        test_array = [
            [
                [-1,  1,  1, -1, -1],
                [-1, -1, -1, -1, -1],
                [ 1,  1, -1, -1, -1],
            ],
            [
                [ 1,  1,  1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1,  1, -1, -1, -1],
            ],
            [
                [-1,  1,  1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1,  1, -1, -1, -1],
            ]
        ]
        for l in range(5):
            assert_array_equal(
                simulation.LTFArray.transform_id(test_array, l),
                test_array
            )


class TestLTFArray(unittest.TestCase):

    test_set = [
        # n, k, mu, sigma
        (64, 4, 0, 1),
        (4, 16, 0, 1),
        (16, 4, 10, .5),
    ]

    def test_random_weights(self):
        for test_parameters in self.test_set:
            n = test_parameters[0]
            k = test_parameters[1]
            mu = test_parameters[2]
            sigma = test_parameters[3]
            self.assertTupleEqual(shape(simulation.LTFArray.normal_weights(n, k, mu, sigma)), (k, n))

    def test_ltf_eval(self):
        """
        Test ltf_eval for correct evaluation of LTFs.
        This is a probabilistic test, relying on random input.
        """

        N = 100 # number of random inputs per test set

        def ltf_eval_slow(x, w):
            """
            evaluate a single input x with a single ltf specified by weights w
            """
            assert len(x) == len(w)
            return dot(x, w)

        for test_parameters in self.test_set:
            n = test_parameters[0]
            k = test_parameters[1]
            mu = test_parameters[2]
            sigma = test_parameters[3]

            inputs = random.choice([-1,+1], (N, n))

            ltf_array = simulation.LTFArray(
                weight_array=simulation.LTFArray.normal_weights(n, k, mu, sigma),
                transform=simulation.LTFArray.transform_id,
                combiner=simulation.LTFArray.combiner_xor,
            )

            fast_evaluation_result = ltf_array.ltf_eval(inputs)
            slow_evaluation_result = []
            for x in inputs:
                slow_evaluation_result.append(
                    [ ltf_eval_slow(x, ltf_array.weight_array[l]) for l in range(k) ]
                )

            self.assertTupleEqual(shape(slow_evaluation_result), (N, k))
            self.assertTupleEqual(shape(fast_evaluation_result), (N, k))
            assert_array_equal(slow_evaluation_result, fast_evaluation_result)
