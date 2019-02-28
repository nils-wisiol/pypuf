"""
This module is used to test the different simulation models.
"""

import unittest
from test.utility import get_functions_with_prefix
from numpy.testing import assert_array_equal
from numpy import shape, dot, array, around, array_equal, reshape, zeros
from numpy.random import RandomState
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray, SimulationMajorityLTFArray
from pypuf import tools


class TestCombiner(unittest.TestCase):
    """This class tests the different combiner functions with predefined input and outputs."""
    def test_combine_xor(self):
        """This function tests the xor combiner function with one pair of input and output."""
        assert_array_equal(
            LTFArray.combiner_xor(
                [
                    [1., 1., -3., 1.],
                    [-1., -1., -1., 1.],
                    [-2., -2., 2., 1.]
                ]
            ),
            [
                -3.,
                -1.,
                8.
            ]
        )

    def test_combine_ip_mod2(self):
        """This function tests the inner product mod 2 combiner function with two pairs of input and output."""
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
    """This class tests the different functions used to transform the input of a LTFArray simulation."""
    def test_id(self):
        """
        This method test the identity function for the predefined inputs (challenges). If every challenge is duplicated
        k times the function works correct.
        """
        test_challenges = array([
            [
                [-1, 1, 1, -1, -1],
                [-1, -1, -1, -1, -1],
                [1, 1, -1, -1, -1],
            ],
            [
                [1, 1, 1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, 1, -1, -1, -1],
            ],
            [
                [-1, 1, 1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, 1, -1, -1, -1],
            ]
        ], dtype=tools.BIT_TYPE)
        for challenges in test_challenges:
            assert_array_equal(
                LTFArray.transform_id(challenges, k=4),
                [
                    [
                        challenges[0],
                        challenges[0],
                        challenges[0],
                        challenges[0]
                    ],
                    [
                        challenges[1],
                        challenges[1],
                        challenges[1],
                        challenges[1]
                    ],
                    [
                        challenges[2],
                        challenges[2],
                        challenges[2],
                        challenges[2]
                    ]
                ]
            )

    def test_atf(self):
        """This method tests the atf transformation with predefined input and output."""
        test_array = array([
            [-1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1],
            [1, 1, -1, -1, -1],
            [1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, 1, -1, -1, -1],
        ], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.transform_atf(test_array, k=3),
            [
                [
                    [-1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, -1],
                ],
                [
                    [-1, 1, -1, 1, -1],
                    [-1, 1, -1, 1, -1],
                    [-1, 1, -1, 1, -1],
                ],
                [
                    [-1, -1, -1, 1, -1],
                    [-1, -1, -1, 1, -1],
                    [-1, -1, -1, 1, -1],
                ],
                [
                    [1, 1, 1, 1, -1],
                    [1, 1, 1, 1, -1],
                    [1, 1, 1, 1, -1],
                ],
                [
                    [-1, 1, -1, 1, -1],
                    [-1, 1, -1, 1, -1],
                    [-1, 1, -1, 1, -1],
                ],
                [
                    [1, -1, -1, 1, -1],
                    [1, -1, -1, 1, -1],
                    [1, -1, -1, 1, -1],
                ]
            ]
        )

    def test_shift(self):
        """This method tests the shift transformation with predefined input and output."""
        test_array = array([
            [-1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1],
            [1, 1, -1, -1, -1],
            [1, 1, -1, -1, -1],
            [1, 2, 3, 4, 5],
        ], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.transform_shift(test_array, k=3),
            [
                [
                    [-1, 1, 1, -1, -1],
                    [1, 1, -1, -1, -1],
                    [1, -1, -1, -1, 1],
                ],
                [
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                ],
                [
                    [1, 1, -1, -1, -1],
                    [1, -1, -1, -1, 1],
                    [-1, -1, -1, 1, 1],
                ],
                [
                    [1, 1, -1, -1, -1],
                    [1, -1, -1, -1, 1],
                    [-1, -1, -1, 1, 1],
                ],
                [
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 1],
                    [3, 4, 5, 1, 2],
                ]
            ]
        )

    def test_secure_lightweight(self):
        """This method tests the secure lightweight transformation with predefined input and output."""
        test_array = array([
            [1, -1, -1, 1, -1, 1],
            [-1, 1, 1, -1, -1, 1],
        ], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.transform_lightweight_secure(test_array, k=3),
            [
                [
                    [1, -1, 1, -1, -1, -1],
                    [1, 1, -1, -1, 1, -1],
                    [-1, 1, -1, 1, -1, 1],
                ],
                [
                    [1, -1, 1, -1, 1, 1],
                    [-1, -1, -1, 1, 1, -1],
                    [1, -1, 1, -1, -1, -1],
                ],
            ]
        )

    def test_att(self):
        """Test the ATT by providing a simple input-output sample."""
        test_array = array([[
            [1, -1, -1, 1, -1, 1],
            [-1, 1, 1, -1, -1, 1],
        ]], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.att(test_array),
            [[
                [-1, -1, 1, -1, -1, 1],
                [-1, 1, 1, 1, -1, 1],
            ]]
        )

    def test_att_inverse(self):
        """Test the inverse ATT by providing a simple input-output sample."""
        test_array = array([[
            [-1, -1, 1, -1, -1, 1],
            [-1, 1, 1, 1, -1, 1],
        ]], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.att_inverse(test_array),
            [[
                [1, -1, -1, 1, -1, 1],
                [-1, 1, 1, -1, -1, 1],
            ]]
        )

    def test_generate_stacked_transform(self):
        """
        This method tests the stacked transformation generation of identity and shift with predefined input and output.
        """
        test_array = array([
            [1, -1, 1, -1],
            [-1, -1, 1, -1],
        ], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.generate_stacked_transform(
                transform_1=LTFArray.transform_id,
                puf_count=2,
                transform_2=LTFArray.transform_shift
            )(test_array, k=4),
            [
                [
                    [1, -1, 1, -1],
                    [1, -1, 1, -1],
                    [1, -1, 1, -1],
                    [-1, 1, -1, 1],
                ],
                [
                    [-1, -1, 1, -1],
                    [-1, -1, 1, -1],
                    [-1, -1, 1, -1],
                    [-1, 1, -1, -1],
                ],
            ]
        )

    def test_generate_random_permutations(self):
        """This method tests the random permutation transformation generation ith predefined inputs and outputs."""
        test_array = array([
            [1, 2, 3, 4],
            [5, 6, 2, 1],
        ], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.generate_random_permutation_transform(
                seed=0xbeef,
                nn=4,
                kk=3,
            )(test_array, k=3),
            [
                [
                    [4, 1, 2, 3],
                    [1, 4, 3, 2],
                    [3, 1, 4, 2],
                ],
                [
                    [1, 5, 6, 2],
                    [5, 1, 2, 6],
                    [2, 5, 1, 6],
                ],
            ],
        )
        assert_array_equal(
            LTFArray.generate_random_permutation_transform(
                seed=0xbeef,
                nn=4,
                kk=3,
                atf=True,
            )(test_array, k=3),
            [
                [
                    [4 * 1 * 2 * 3, 1 * 2 * 3, 2 * 3, 3],
                    [1 * 4 * 3 * 2, 4 * 3 * 2, 3 * 2, 2],
                    [3 * 1 * 4 * 2, 1 * 4 * 2, 4 * 2, 2],
                ],
                [
                    [1 * 5 * 6 * 2, 5 * 6 * 2, 6 * 2, 2],
                    [5 * 1 * 2 * 6, 1 * 2 * 6, 2 * 6, 6],
                    [2 * 5 * 1 * 6, 5 * 1 * 6, 1 * 6, 6],
                ],
            ],
        )

    def test_generate_concatenated_transform(self):
        """
        This method tests the concatenation of 1 to n bent and identity transformation with predefined input and output.
        """
        test_array = array([
            [1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1],
        ], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.generate_concatenated_transform(
                transform_1=LTFArray.transform_id,
                bit_number_transform_1=6,
                transform_2=LTFArray.transform_atf,
            )(test_array, k=3),
            [
                [
                    [1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1],
                    [1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1],
                    [1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1],
                ],
            ]
        )

    def test_polynomial(self):
        """This method tests the polynomial transformation with predefined input and output for k=4 PUFs."""
        test_array = array([
            [-1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1,
             1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1,
             1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1,
             1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1]
        ], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.transform_polynomial(test_array, k=4),
            [
                [
                    [-1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1,
                     -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1,
                     -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1],
                    [-1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1,
                     -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1,
                     -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1,
                     1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1,
                     -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1],
                    [1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                     -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1,
                     1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1]
                ]
            ]
        )

    def test_polynomial_k1(self):
        """This method tests the polynomial transformation with predefined input and output for k=1 PUF."""
        test_array = array([
            [-1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1,
             1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1,
             1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1,
             1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1]
        ], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.transform_polynomial(test_array, k=1),
            [
                [
                    [-1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1,
                     -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1,
                     -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1]
                ]
            ]
        )

    def test_permutation_atf(self):
        """This method tests the permuation atf transformation with predefined input and output for k=4 PUFs."""
        test_array = array([
            [1, -1, -1, 1, -1, 1],
            [-1, 1, 1, -1, -1, 1],
        ], dtype=tools.BIT_TYPE)
        assert_array_equal(
            LTFArray.transform_permutation_atf(test_array, k=4),
            [
                [
                    [-1, -1, 1, -1, -1, 1],
                    [-1, 1, -1, -1, -1, -1],
                    [-1, 1, 1, 1, -1, 1],
                    [-1, 1, 1, -1, -1, -1]
                ],

                [
                    [-1, -1, -1, -1, 1, -1],
                    [-1, -1, 1, 1, -1, 1],
                    [-1, 1, -1, 1, 1, 1],
                    [-1, -1, -1, 1, -1, 1]
                ]
            ]
        )


class TestLTFArray(unittest.TestCase):
    """
    This class is used to test the LTFArray class. For this purpose this class provides a set of instance parameter
    in oder to test several configurations.
    """
    test_set = [
        # n, k, mu, sigma, bias
        (64, 4, 0, 1, None),
        (4, 16, 0, 1, None),
        (16, 4, 10, .5, None),
        (64, 4, 0, 1, 1.5),
        (4, 16, 0, 1, 2.3),
        (16, 4, 10, .5, array([1.4, 1.0, 0.5, 0.4])),
    ]

    def test_bias_influence_value(self):
        """
        This method tests the influence of the bias value. The results should be different.
        """
        n = 8
        k = 4
        mu = 1
        sigma = 0.5

        challenges = tools.all_inputs(n)

        weight_array = LTFArray.normal_weights(n, k, mu=mu, sigma=sigma, random_instance=RandomState(0xBADA556))
        bias_value = 2.5

        biased_ltf_array = LTFArray(
            weight_array=weight_array,
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            bias=bias_value,
        )
        ltf_array = LTFArray(
            weight_array=weight_array,
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            bias=None,
        )
        # the second dimension of the weight_array must be the number of elements in biased weight_array
        self.assertEqual(shape(ltf_array.weight_array)[1], shape(biased_ltf_array.weight_array)[1])

        bias_array = biased_ltf_array.weight_array[:, -1]
        bias_array_compared = [bias == bias_array[0] for bias in bias_array]
        # the bias values should be equal for this test.
        self.assertTrue(array(bias_array_compared).all())

        biased_responses = biased_ltf_array.eval(challenges)
        responses = ltf_array.eval(challenges)

        self.assertFalse(array_equal(biased_responses, responses))

    def test_bias_influence_array(self):
        """
        This method tests the influence of the bias array. The results should be different.
        """
        n = 8
        k = 4
        mu = 1
        sigma = 0.5

        challenges = tools.all_inputs(n)

        weight_array = LTFArray.normal_weights(n, k, mu=mu, sigma=sigma, random_instance=RandomState(0xBADA556))
        bias_array = LTFArray.normal_weights(1, k, mu=mu, sigma=sigma*2, random_instance=RandomState(0xBADAFF1))

        biased_ltf_array = LTFArray(
            weight_array=weight_array,
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            bias=bias_array,
        )
        ltf_array = LTFArray(
            weight_array=weight_array,
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            bias=None,
        )
        # the second dimension of the weight_array must be the number of elements in biased weight_array
        self.assertEqual(shape(ltf_array.weight_array)[1], shape(biased_ltf_array.weight_array)[1])
        assert_array_equal(biased_ltf_array.weight_array[:, :n], ltf_array.weight_array[:, :n])
        assert_array_equal(biased_ltf_array.weight_array[:, :n], weight_array)
        assert_array_equal(biased_ltf_array.weight_array[:, n], reshape(bias_array, (k,)))
        assert_array_equal(ltf_array.weight_array[:, n], zeros((k, )))

        biased_responses = biased_ltf_array.eval(challenges)
        responses = ltf_array.eval(challenges)

        self.assertFalse(array_equal(biased_responses, responses))

    def test_random_weights(self):
        """
        This method tests if the shape of the LTFArray generated weights are as expected.
        """
        for test_parameters in self.test_set:
            n = test_parameters[0]
            k = test_parameters[1]
            mu = test_parameters[2]
            sigma = test_parameters[3]
            self.assertTupleEqual(shape(LTFArray.normal_weights(n, k, mu, sigma)), (k, n))

    def test_ltf_eval(self):
        """
        Test ltf_eval for correct evaluation of LTFs.
        """

        N = 100  # number of random inputs per test set

        def ltf_eval_slow(challenge, weights):
            """
            evaluate a single challenge with a single ltf specified by weights
            """
            assert len(challenge) == len(weights) - 1
            return dot(challenge, weights[:n]) + weights[n]  # weights[n] is the bias

        for test_parameters in self.test_set:
            n = test_parameters[0]
            k = test_parameters[1]
            mu = test_parameters[2]
            sigma = test_parameters[3]

            inputs = tools.random_inputs(n, N, random_instance=RandomState(0xA1))

            ltf_array = LTFArray(
                weight_array=LTFArray.normal_weights(n, k, mu, sigma),
                transform=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
            )

            fast_evaluation_result = around(ltf_array.ltf_eval(LTFArray.transform_id(inputs, k)), decimals=8)
            slow_evaluation_result = []
            for challenge in inputs:
                slow_evaluation_result.append(
                    [ltf_eval_slow(challenge, ltf_array.weight_array[l]) for l in range(k)]
                )
            slow_evaluation_result = around(slow_evaluation_result, decimals=8)

            self.assertTupleEqual(shape(slow_evaluation_result), (N, k))
            self.assertTupleEqual(shape(fast_evaluation_result), (N, k))
            assert_array_equal(slow_evaluation_result, fast_evaluation_result)


class TestNoisyLTFArray(TestLTFArray):
    """This class is used to test the NoisyLTFArray class."""
    def test_ltf_eval(self):
        """
        Test ltf_eval for correct evaluation of LTFs.
        """

        weight_prng_1 = RandomState(seed=0xBADA55)
        weight_prng_2 = RandomState(seed=0xBADA55)
        noise_prng_1 = RandomState(seed=0xC0FFEE)
        noise_prng_2 = RandomState(seed=0xC0FFEE)

        N = 100  # number of random inputs per test set

        for test_parameters in self.test_set:
            n = test_parameters[0]
            k = test_parameters[1]
            mu = test_parameters[2]
            sigma = test_parameters[3]

            transformed_inputs = LTFArray.transform_id(
                tools.random_inputs(n, N, random_instance=RandomState(seed=0xBAADA555)),
                k
            )

            ltf_array = LTFArray(
                weight_array=LTFArray.normal_weights(n, k, mu, sigma, weight_prng_1),
                transform=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
            )

            noisy_ltf_array = NoisyLTFArray(
                weight_array=LTFArray.normal_weights(n, k, mu, sigma, weight_prng_2),
                # weight_prng_2 was seeded identically to weight_prng_1
                transform=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
                sigma_noise=1,
                random_instance=noise_prng_1,
            )

            evaled_ltf_array = ltf_array.ltf_eval(transformed_inputs)
            assert_array_equal(
                around(evaled_ltf_array + noise_prng_2.normal(loc=0, scale=1,
                                                              size=(len(evaled_ltf_array), k)), decimals=10),
                around(noisy_ltf_array.ltf_eval(transformed_inputs), decimals=10)
            )

    def test_bias_influence_array(self):
        """
        This method tests the influence of the bias array. The results should be different.
        """
        n = 8
        k = 4
        mu = 1
        sigma = 0.5

        challenges = tools.all_inputs(n)

        weight_array = NoisyLTFArray.normal_weights(n, k, mu=mu, sigma=sigma, random_instance=RandomState(0xBADA556))
        bias_array = NoisyLTFArray.normal_weights(1, k, mu=mu, sigma=sigma * 2, random_instance=RandomState(0xBADAFF1))

        biased_ltf_array = NoisyLTFArray(
            weight_array=weight_array,
            transform=NoisyLTFArray.transform_id,
            combiner=NoisyLTFArray.combiner_xor,
            sigma_noise=sigma,
            bias=bias_array,
        )
        ltf_array = NoisyLTFArray(
            weight_array=weight_array,
            transform=NoisyLTFArray.transform_id,
            combiner=NoisyLTFArray.combiner_xor,
            sigma_noise=sigma,
            bias=None,
        )
        self.assertEqual(ltf_array.weight_array.shape, biased_ltf_array.weight_array.shape)

        bias_array = biased_ltf_array.weight_array[:, -1]
        bias_array_compared = [bias == bias_array[0] for bias in bias_array[1:]]
        # the bias values should be different for this test. It is possible that they are all equal but this chance is
        # low.
        self.assertFalse(array(bias_array_compared).all())

        biased_responses = biased_ltf_array.eval(challenges)
        responses = ltf_array.eval(challenges)

        # The arithmetic mean of the res
        self.assertFalse(array_equal(biased_responses, responses))

    def test_bias_influence_value(self):
        """
        This method tests the influence of the bias value. The results should be different.
        """
        n = 8
        k = 4
        mu = 1
        sigma = 0.5

        challenges = tools.all_inputs(n)

        weight_array = NoisyLTFArray.normal_weights(n, k, mu=mu, sigma=sigma, random_instance=RandomState(0xBADA556))
        bias_value = 2.5

        biased_ltf_array = NoisyLTFArray(
            weight_array=weight_array,
            transform=NoisyLTFArray.transform_id,
            combiner=NoisyLTFArray.combiner_xor,
            sigma_noise=sigma,
            bias=bias_value,
        )
        ltf_array = NoisyLTFArray(
            weight_array=weight_array,
            transform=NoisyLTFArray.transform_id,
            combiner=NoisyLTFArray.combiner_xor,
            sigma_noise=sigma,
            bias=None,
        )
        self.assertEqual(ltf_array.weight_array.shape, biased_ltf_array.weight_array.shape)

        bias_array = biased_ltf_array.weight_array[:, -1]
        bias_array_compared = [bias == bias_array[0] for bias in bias_array]
        # the bias values should be equal for this test.
        self.assertTrue(array(list(bias_array_compared)).all())

        biased_responses = biased_ltf_array.eval(challenges)
        responses = ltf_array.eval(challenges)

        # The arithmetic mean of the res
        self.assertFalse(array_equal(biased_responses, responses))

    def test_init_normal_empirical(self):
        """
        Test if initialization by intra distance yields the desired intra distance.
        """
        for intra_dist in [.1, .2, .3]:
            nla = NoisyLTFArray.init_normal_empirical(32, 1, NoisyLTFArray.transform_id, NoisyLTFArray.combiner_xor,
                                                      intra_dist, approx_threshold=.01,
                                                      random_instance=RandomState(0xbeef))
            self.assertTrue(abs(tools.approx_dist(nla, nla, 10000) - intra_dist) < .02)
        for intra_dist in [.1, .2, .3]:
            nla = NoisyLTFArray.init_normal_empirical(64, 4, NoisyLTFArray.transform_id, NoisyLTFArray.combiner_xor,
                                                      intra_dist, approx_threshold=.1,
                                                      random_instance=RandomState(0xbeef))
            self.assertTrue(abs(tools.approx_dist(nla, nla, 10000) - intra_dist) < .15)


class TestSimulationMajorityLTFArray(unittest.TestCase):
    """This class is used to test the SimulationMajorityLTFArray class."""
    def test_majority_voting(self):
        """This method is used to test if majority vote works.
        The first test checks for unequal PUF responses with a LTFArray without noise and a SimulationMajorityLTFArray
        instance with noise. The second test checks if majority voting works and the instance with and without noise
        generate equal resonses.
        """
        weight_prng1 = RandomState(seed=0xBADA55)
        noise_prng = RandomState(seed=0xC0FFEE)
        crp_count = 16  # number of random inputs per test set
        n = 8
        k = 16
        mu = 0
        sigma = 1
        vote_count = 1

        weight_array = LTFArray.normal_weights(n, k, mu, sigma, weight_prng1)

        mv_noisy_ltf_array = SimulationMajorityLTFArray(
            weight_array=weight_array,
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            sigma_noise=1,
            random_instance_noise=noise_prng,
            vote_count=vote_count,
        )

        ltf_array = LTFArray(
            weight_array=weight_array,
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor
        )

        inputs = tools.random_inputs(n, crp_count, random_instance=RandomState(seed=0xDEADDA7A))

        ltf_array_result = ltf_array.eval(inputs)
        mv_noisy_ltf_array_result = mv_noisy_ltf_array.eval(inputs)
        # These test checks if the output is different because of noise
        self.assertFalse(array_equal(ltf_array_result, mv_noisy_ltf_array_result), 'These arrays must be different')

        # reset pseudo random number generator
        noise_prng = RandomState(seed=0xC0FFEE)
        # increase the vote_count in order to get equality
        vote_count = 2845
        mv_noisy_ltf_array = SimulationMajorityLTFArray(
            weight_array=weight_array,
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            sigma_noise=1,
            random_instance_noise=noise_prng,
            vote_count=vote_count,
        )
        # This checks if the majority vote works
        mv_noisy_ltf_array_result = mv_noisy_ltf_array.eval(inputs)
        assert_array_equal(mv_noisy_ltf_array_result, ltf_array_result)

    def test_transformations_combiner(self):
        """
        This test checks all combinations of transformations and combiners for SimulationMajorityLTFArray to run.
        """
        noise_prng = RandomState(seed=0xC0FFEE)
        weight_prng1 = RandomState(seed=0xBADA55)
        crp_count = 16  # number of random inputs per test set
        n = 64
        k = 2
        mu = 0
        sigma = 1
        vote_count = 1
        weight_array = LTFArray.normal_weights(n, k, mu, sigma, weight_prng1)

        inputs = tools.random_inputs(n, crp_count, random_instance=RandomState(seed=0xDEADDA7A))

        combiners = get_functions_with_prefix('combiner_', SimulationMajorityLTFArray)
        transformations = get_functions_with_prefix('transform_', SimulationMajorityLTFArray)

        for transformation in transformations:
            for combiner in combiners:
                mv_noisy_ltf_array = SimulationMajorityLTFArray(
                    weight_array=weight_array,
                    transform=transformation,
                    combiner=combiner,
                    sigma_noise=1,
                    random_instance_noise=noise_prng,
                    vote_count=vote_count,
                )
                mv_noisy_ltf_array.eval(inputs)

    def test_bias_influence(self):
        """
        This method tests the influence of the bias array. The results should be different.
        """
        n = 8
        k = 4
        mu = 1
        sigma = 0.5

        challenges = tools.all_inputs(n)

        weight_array = SimulationMajorityLTFArray.normal_weights(n, k, mu=mu, sigma=sigma,
                                                                 random_instance=RandomState(0xBADA556))
        bias_array = SimulationMajorityLTFArray.normal_weights(1, k, mu=mu, sigma=sigma * 2,
                                                               random_instance=RandomState(0xBADAFF1))

        biased_ltf_array = SimulationMajorityLTFArray(
            weight_array=weight_array,
            transform=SimulationMajorityLTFArray.transform_id,
            combiner=SimulationMajorityLTFArray.combiner_xor,
            sigma_noise=sigma,
            random_instance_noise=RandomState(0xCCABAD),
            bias=bias_array,
        )
        ltf_array = SimulationMajorityLTFArray(
            weight_array=weight_array,
            transform=SimulationMajorityLTFArray.transform_id,
            combiner=SimulationMajorityLTFArray.combiner_xor,
            sigma_noise=sigma,
            random_instance_noise=RandomState(0xCCABAD),
            bias=None,
        )
        self.assertEqual(ltf_array.weight_array.shape, biased_ltf_array.weight_array.shape)

        bias_array = biased_ltf_array.weight_array[:, -1]
        bias_array_compared = [bias == bias_array[0] for bias in bias_array[1:]]
        # the bias values should be different for this test. It is possible that they are all equal but this chance is
        # low.
        self.assertFalse(array(bias_array_compared).all())

        biased_responses = biased_ltf_array.eval(challenges)
        responses = ltf_array.eval(challenges)

        # The arithmetic mean of the res
        self.assertFalse(array_equal(biased_responses, responses))

    def test_bias_influence_value(self):
        """
        This method tests the influence of the bias value. The results should be different.
        """
        n = 8
        k = 4
        mu = 1
        sigma = 0.5

        challenges = tools.all_inputs(n)

        weight_array = SimulationMajorityLTFArray.normal_weights(n, k, mu=mu, sigma=sigma,
                                                                 random_instance=RandomState(0xBADA556))
        bias_value = 2.5

        biased_ltf_array = SimulationMajorityLTFArray(
            weight_array=weight_array,
            transform=SimulationMajorityLTFArray.transform_id,
            combiner=SimulationMajorityLTFArray.combiner_xor,
            sigma_noise=sigma,
            random_instance_noise=RandomState(0xCCABAD),
            bias=bias_value,
        )
        ltf_array = SimulationMajorityLTFArray(
            weight_array=weight_array,
            transform=SimulationMajorityLTFArray.transform_id,
            combiner=SimulationMajorityLTFArray.combiner_xor,
            sigma_noise=sigma,
            random_instance_noise=RandomState(0xCCABAD),
            bias=None,
        )
        self.assertEqual(ltf_array.weight_array.shape, biased_ltf_array.weight_array.shape)

        bias_array = biased_ltf_array.weight_array[:, -1]
        bias_array_compared = [bias == bias_array[0] for bias in bias_array]
        # the bias values should be equal for this tests.
        self.assertTrue(array(bias_array_compared).all())

        biased_responses = biased_ltf_array.eval(challenges)
        responses = ltf_array.eval(challenges)
        self.assertFalse(array_equal(biased_responses, responses))
