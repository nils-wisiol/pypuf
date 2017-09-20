"""
This module is used to test the different simulation models.
"""

import unittest
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray, SimulationMajorityLTFArray
from pypuf import tools
from numpy.testing import assert_array_equal
from numpy import shape, dot, array, around, array_equal
from numpy.random import RandomState


class TestCombiner(unittest.TestCase):
    """This class tests the different combiner functions with predefined input and outputs."""
    def test_combine_xor(self):
        """This function tests the xor combiner function with one pair of input and output."""
        assert_array_equal(
            LTFArray.combiner_xor(
                [
                    [1, 1, -3, 1],
                    [-1, -1, -1, 1],
                    [-2, -2, 2, 1]
                ]
            ),
            [
                -3,
                -1,
                8
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
        test_challnges = [
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
        ]
        for challenges in test_challnges:
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
        ])
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
        ])
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
        ])
        assert_array_equal(
            LTFArray.transform_lightweight_secure(test_array, k=3),
            [
                [
                    [-1, -1, -1, 1, 1, -1],
                    [-1, -1, 1, 1, -1, -1],
                    [-1, 1, 1, -1, -1, -1],
                ],
                [
                    [-1, -1, -1, -1, 1, 1],
                    [-1, -1, -1, 1, 1, -1],
                    [-1, -1, 1, 1, -1, -1],
                ],
            ]
        )

    def test_shift_secure_lightweight(self):
        """This method tests the shift secure lightweight transformation with predefined input and output."""
        test_array = array([
            [1, -1, -1, 1, -1, 1],
        ])
        assert_array_equal(
            LTFArray.transform_shift_lightweight_secure(test_array, k=3),
            [
                [
                    [-1, -1, -1, 1, 1, -1],
                    [1, -1, 1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, 1],
                ],
            ]
        )

    def test_1_n_bent(self):
        """This method tests the one to n bent transformation with predefined inputs and outputs."""
        test_array = array([
            [1, -1, -1, 1, -1, 1],
            [-1, 1, 1, -1, -1, 1],
        ])
        assert_array_equal(
            LTFArray.transform_1_n_bent(test_array, k=3),
            [
                [
                    [1, -1, 1, -1, 1, -1],
                    [1, -1, -1, 1, -1, 1],
                    [1, -1, -1, 1, -1, 1],
                ],
                [
                    [1, -1, 1, -1, 1, -1],
                    [-1, 1, 1, -1, -1, 1],
                    [-1, 1, 1, -1, -1, 1],
                ],
            ]
        )
        test_array = array([
            [1, -1, -1, 1, -1, 1],
            [-1, 1, 1, -1, -1, 1],
        ])
        assert_array_equal(
            LTFArray.transform_1_n_bent(test_array, k=1),
            [
                [
                    [1, -1, 1, -1, 1, -1],
                ],
                [
                    [1, -1, 1, -1, 1, -1],
                ],
            ]
        )

    def test_generate_stacked_transform(self):
        """
        This method tests the stacked transformation generation of identity and shift with predefined input and output.
        """
        test_array = array([
            [1, -1, 1, -1],
            [-1, -1, 1, -1],
        ])
        assert_array_equal(
            LTFArray.generate_stacked_transform(
                transform_1=LTFArray.transform_id,
                kk=2,
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
            [10, 20, 30, 40],
        ])
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
                    [40, 10, 20, 30],
                    [10, 40, 30, 20],
                    [30, 10, 40, 20],
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
                    [40 * 10 * 20 * 30, 10 * 20 * 30, 20 * 30, 30],
                    [10 * 40 * 30 * 20, 40 * 30 * 20, 30 * 20, 20],
                    [30 * 10 * 40 * 20, 10 * 40 * 20, 40 * 20, 20],
                ],
            ],
        )

    def test_generate_concatenated_transform(self):
        """
        This method tests the concatenation of 1 to n bent and identity transformation with predefined input and output.
        """
        test_array = array([
            [1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1],
        ])
        assert_array_equal(
            LTFArray.generate_concatenated_transform(
                transform_1=LTFArray.transform_1_n_bent,
                nn=6,
                transform_2=LTFArray.transform_id,
            )(test_array, k=3),
            [
                [
                    [1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1],
                    [1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1],
                    [1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1],
                ],
            ]
        )

    def test_1_1_bent(self):
        """This method test the one to one bent transformation with predefined input and output."""
        test_array = array([
            [1, -1, -1, 1, -1, 1],
            [-1, 1, 1, -1, -1, 1],
        ])
        assert_array_equal(
            LTFArray.transform_1_1_bent(test_array, k=3),
            [
                [
                    [1, -1, -1, 1, -1, 1],
                    [1, -1, -1, 1, -1, 1],
                    [1, -1, -1, 1, -1, 1],
                ],
                [
                    [1, 1, 1, -1, -1, 1],
                    [-1, 1, 1, -1, -1, 1],
                    [-1, 1, 1, -1, -1, 1],
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
        ])
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
        ])
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
        ])
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
            weight_array = LTFArray.normal_weights(n, k, mu, sigma)

            input_len = n - 1 if bias else n
            inputs = RandomState(seed=0xBAADA555).choice([-1, +1], (N, input_len))  # bad ass testing

            biased_ltf_array = LTFArray(
                weight_array=weight_array,
                transform=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
                bias=bias,
            )
            ltf_array = LTFArray(
                weight_array=weight_array,
                transform=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
                bias=False,
            )
            biased_eval = biased_ltf_array.eval(inputs)
            inputs = tools.iter_append_last(inputs, 1) \
                if biased_ltf_array.bias else inputs
            ltf_array_evaled_output = ltf_array.eval(inputs)

            assert_array_equal(biased_eval, ltf_array_evaled_output)

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
            assert len(challenge) == len(weights)
            return dot(challenge, weights)

        for test_parameters in self.test_set:
            n = test_parameters[0]
            k = test_parameters[1]
            mu = test_parameters[2]
            sigma = test_parameters[3]

            inputs = RandomState(seed=0xCAFED00D).choice([-1, +1], (N, n))

            ltf_array = LTFArray(
                weight_array=LTFArray.normal_weights(n, k, mu, sigma),
                transform=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
            )

            fast_evaluation_result = around(ltf_array.ltf_eval(LTFArray.transform_id(inputs, k)), decimals=10)
            slow_evaluation_result = []
            for challenge in inputs:
                slow_evaluation_result.append(
                    [ltf_eval_slow(challenge, ltf_array.weight_array[l]) for l in range(k)]
                )
            slow_evaluation_result = around(slow_evaluation_result, decimals=10)

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
                RandomState(seed=0xBAADA555).choice([-1, +1], (N, n)),  # bad ass testing
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

        inputs = array(list(tools.random_inputs(n, crp_count, random_instance=RandomState(seed=0xDEADDA7A))))

        ltf_array_result = ltf_array.eval(inputs)
        mv_noisy_ltf_array_result = mv_noisy_ltf_array.eval(inputs)
        # These test checks if the output is different because of noise
        self.assertFalse(array_equal(ltf_array_result, mv_noisy_ltf_array_result), 'These arrays must be different')

        # reset pseudo random number generator
        noise_prng = RandomState(seed=0xC0FFEE)
        # increase the vote_count in order to get equality
        vote_count = 277
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
        n = 8
        k = 2
        mu = 0
        sigma = 1
        vote_count = 1
        weight_array = LTFArray.normal_weights(n, k, mu, sigma, weight_prng1)

        inputs = array(list(tools.random_inputs(n, crp_count, random_instance=RandomState(seed=0xDEADDA7A))))

        def get_functions_with_prefix(prefix, obj):
            """
            This function return all functions with a prefix.
            :param prefix: string
            :param obj: object
                        Object to investigate
            :return list of strings
            """
            return [func for func in dir(obj) if func.startswith(prefix)]

        combiners = get_functions_with_prefix('combiner_', SimulationMajorityLTFArray)
        transformations = get_functions_with_prefix('transform_', SimulationMajorityLTFArray)

        for transformation in transformations:
            for combiner in combiners:
                mv_noisy_ltf_array = SimulationMajorityLTFArray(
                    weight_array=weight_array,
                    transform=getattr(SimulationMajorityLTFArray, transformation),
                    combiner=getattr(SimulationMajorityLTFArray, combiner),
                    sigma_noise=1,
                    random_instance_noise=noise_prng,
                    vote_count=vote_count,
                )
                mv_noisy_ltf_array.eval(inputs)
