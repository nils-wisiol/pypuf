import unittest
from pypuf import tools
from pypuf.simulation.ltfarray import LTFArray
from numpy.testing import assert_array_equal
from numpy import shape, dot, random, full, tile, array, transpose, abs, around


class TestCombiner(unittest.TestCase):

    def test_combine_xor(self):
        assert_array_equal(
            LTFArray.combiner_xor(
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

    def test_combine_ip_mod2(self):
        assert_array_equal(
            LTFArray.combiner_ip_mod2(
                array([
                    [1, 1, -3, 1],
                    [-1, -1, -1, 1],
                    [-2, -2, 2, 1]
                ])
            ),
            [
                1,
                -1,
                -4
            ]
        )
        assert_array_equal(
            LTFArray.combiner_ip_mod2(
                array([
                    [1, 1, 1, 1, 1, 1],
                    [-1, -1, -1, 1, -1, -1],
                    [-2, -2, 2, 1, 10, 10]
                ])
            ),
            [
                1,
                1,
                -40
            ]
        )


class TestInputTransformation(unittest.TestCase):

    def test_id(self):
        test_cs = [
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
        for cs in test_cs:
            assert_array_equal(
                LTFArray.transform_id(cs, k=4),
                [
                    [
                        cs[0],
                        cs[0],
                        cs[0],
                        cs[0]
                    ],
                    [
                        cs[1],
                        cs[1],
                        cs[1],
                        cs[1]
                    ],
                    [
                        cs[2],
                        cs[2],
                        cs[2],
                        cs[2]
                    ]
                ]
            )

    def test_atf(self):
        test_array = array([
                [-1,  1,  1, -1, -1],
                [-1, -1, -1, -1, -1],
                [ 1,  1, -1, -1, -1],
                [ 1,  1,  1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1,  1, -1, -1, -1],
        ])
        assert_array_equal(
            LTFArray.transform_atf(test_array, k=3),
            [
                [
                    [-1,  1,  1,  1, -1],
                    [-1,  1,  1,  1, -1],
                    [-1,  1,  1,  1, -1],
                ],
                [
                    [-1,  1, -1,  1, -1],
                    [-1,  1, -1,  1, -1],
                    [-1,  1, -1,  1, -1],
                ],
                [
                    [-1, -1, -1,  1, -1],
                    [-1, -1, -1,  1, -1],
                    [-1, -1, -1,  1, -1],
                ],
                [
                    [ 1,  1,  1,  1, -1],
                    [ 1,  1,  1,  1, -1],
                    [ 1,  1,  1,  1, -1],
                ],
                [
                    [-1,  1, -1,  1, -1],
                    [-1,  1, -1,  1, -1],
                    [-1,  1, -1,  1, -1],
                ],
                [
                    [ 1, -1, -1,  1, -1],
                    [ 1, -1, -1,  1, -1],
                    [ 1, -1, -1,  1, -1],
                ]
            ]
        )

    def test_shift(self):
        test_array = array([
                [-1,  1,  1, -1, -1],
                [-1, -1, -1, -1, -1],
                [ 1,  1, -1, -1, -1],
                [ 1,  1, -1, -1, -1],
                [ 1,  2,  3,  4,  5],
        ])
        assert_array_equal(
            LTFArray.transform_shift(test_array, k=3),
            [
                [
                    [-1,  1,  1, -1, -1],
                    [ 1,  1, -1, -1, -1],
                    [ 1, -1, -1, -1,  1],
                ],
                [
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                ],
                [
                    [ 1,  1, -1, -1, -1],
                    [ 1, -1, -1, -1,  1],
                    [-1, -1, -1,  1,  1],
                ],
                [
                    [ 1,  1, -1, -1, -1],
                    [ 1, -1, -1, -1,  1],
                    [-1, -1, -1,  1,  1],
                ],
                [
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 1],
                    [3, 4, 5, 1, 2],
                ]
            ]
        )

    def test_secure_lightweight(self):
        test_array = array([
            [ 1, -1, -1,  1, -1,  1],
            [-1,  1,  1, -1, -1,  1],
        ])
        assert_array_equal(
            LTFArray.transform_lightweight_secure(test_array, k=3),
            [
                [
                    [-1, -1, -1,  1,  1, -1],
                    [-1, -1,  1,  1, -1, -1],
                    [-1,  1,  1, -1, -1, -1],
                ],
                [
                    [-1, -1, -1, -1,  1,  1],
                    [-1, -1, -1,  1,  1, -1],
                    [-1, -1,  1,  1, -1, -1],
                ],
            ]
        )

    def test_1_n_bent(self):
        test_array = array([
            [ 1, -1, -1,  1, -1,  1],
            [-1,  1,  1, -1, -1,  1],
        ])
        assert_array_equal(
            LTFArray.transform_1_n_bent(test_array, k=3),
            [
                [
                    [ 1, -1,  1, -1,  1, -1],
                    [ 1, -1, -1,  1, -1,  1],
                    [ 1, -1, -1,  1, -1,  1],
                ],
                [
                    [ 1, -1,  1, -1,  1, -1],
                    [-1,  1,  1, -1, -1,  1],
                    [-1,  1,  1, -1, -1,  1],
                ],
            ]
        )

    def test_1_1_bent(self):
        test_array = array([
            [ 1, -1, -1,  1, -1,  1],
            [-1,  1,  1, -1, -1,  1],
        ])
        assert_array_equal(
            LTFArray.transform_1_1_bent(test_array, k=3),
            [
                [
                    [ 1, -1, -1,  1, -1,  1],
                    [ 1, -1, -1,  1, -1,  1],
                    [ 1, -1, -1,  1, -1,  1],
                ],
                [
                    [ 1,  1,  1, -1, -1,  1],
                    [-1,  1,  1, -1, -1,  1],
                    [-1,  1,  1, -1, -1,  1],
                ],
            ]
        )


class TestLTFArray(unittest.TestCase):

    test_set = [
        # n, k, mu, sigma, bias
        (64, 4, 0, 1, False),
        (4, 16, 0, 1, False),
        (16, 4, 10, .5, False),
        (64, 4, 0, 1, True),
        (4, 16, 0, 1, True),
        (16, 4, 10, .5, True),
    ]

    def test_bias(self):
        """
        Probabilistic test for checking the bias feature in eval.
        """
        N = 100

        for test_parameters in self.test_set:
            n = test_parameters[0]
            k = test_parameters[1]
            mu = test_parameters[2]
            sigma = test_parameters[3]
            bias = test_parameters[4]
            weight_array = simulation.LTFArray.normal_weights(n, k, mu, sigma)

            input_len = n-1 if bias else n
            inputs = random.choice([-1,+1], (N, input_len))

            biased_ltf_array = simulation.LTFArray(
                weight_array = weight_array,
                transform = simulation.LTFArray.transform_id,
                combiner = simulation.LTFArray.combiner_xor,
                bias = bias,
            )
            ltf_array = simulation.LTFArray(
                weight_array = weight_array,
                transform = simulation.LTFArray.transform_id,
                combiner = simulation.LTFArray.combiner_xor,
                bias = False,
            )
            biased_eval = biased_ltf_array.eval(inputs)
            inputs = tools.iter_append_last(inputs, 1)\
                if biased_ltf_array.bias else inputs
            eval = ltf_array.eval(inputs)

            assert_array_equal(biased_eval, eval)

    def test_random_weights(self):
        for test_parameters in self.test_set:
            n = test_parameters[0]
            k = test_parameters[1]
            mu = test_parameters[2]
            sigma = test_parameters[3]
            self.assertTupleEqual(shape(LTFArray.normal_weights(n, k, mu, sigma)), (k, n))

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
            random.seed(0xfabfab)
            inputs = random.choice([-1,+1], (N, n))

            ltf_array = LTFArray(
                weight_array=LTFArray.normal_weights(n, k, mu, sigma),
                transform=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
            )

            fast_evaluation_result = around(ltf_array.ltf_eval(LTFArray.transform_id(inputs, k)), decimals=10)
            slow_evaluation_result = []
            for c in inputs:
                slow_evaluation_result.append(
                    [ ltf_eval_slow(c, ltf_array.weight_array[l]) for l in range(k) ]
                )
            slow_evaluation_result = around(slow_evaluation_result, decimals=10)
            self.assertTupleEqual(shape(slow_evaluation_result), (N, k))
            self.assertTupleEqual(shape(fast_evaluation_result), (N, k))
            assert_array_equal(slow_evaluation_result, fast_evaluation_result)
