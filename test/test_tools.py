"""This module is used to test the functions which are implemented in pypuf.tools."""
import unittest
from numpy import zeros, dtype, array_equal, array
from numpy.random import RandomState
from numpy.testing import assert_array_equal
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import random_input, all_inputs, random_inputs, sample_inputs, chi_vectorized, append_last, \
    TrainingSet, RESULT_TYPE, transform_challenge_11_to_01, transform_challenge_01_to_11, poly_mult_div


class TestAppendLast(unittest.TestCase):
    """This class is used to test the append_last method from tools."""

    def test_proper_dimension(self):
        """The function recognizes that the array has dimension zero."""
        arr = zeros((), dtype=dtype('int64'))
        item = 30
        with self.assertRaisesRegex(AssertionError, 'arr must have at least one dimension.'):
            append_last(arr, item)

    def test_append_result(self):
        """This function checks for the correct calculation of the function with predefined results."""
        typ = dtype('int64')
        arr_check = zeros((3, 2, 2), dtype=typ)
        arr = zeros((3, 2, 1), dtype=typ)
        item = 0

        arr_res = append_last(arr, item)
        self.assertTrue(array_equal(arr_check, arr_res), 'The arrays should be equal.')

    def test_append_dimensions(self):
        """This test checks the correct behavior for different shapes."""
        typ = dtype('int64')
        item = 0
        # iterate over several dimensions
        for dimensions in range(1, 11):
            # generate the shape of arr
            shape = [i for i in range(dimensions)]
            # generate the shape of the array which is used to check the adding
            shape_res = [i for i in range(dimensions)]
            # the last dimension must have one more element than the last dimension of arr
            shape_res[-1] = shape_res[-1] + 1
            # generate the arrays
            arr_check = zeros(shape_res, dtype=typ)
            arr = zeros(shape, dtype=typ)
            # append the last item
            arr_res = append_last(arr, item)
            # check for equality
            self.assertTrue(array_equal(arr_res, arr_check), 'The arrays should be equal.')

    def test_training_set_challenges(self):
        """The TrainingSet should generate the same challenges for equal seeds."""
        n = 8
        k = 1
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        N = 1000
        instance_prng = RandomState(0x4EFEA)
        weight_array = LTFArray.normal_weights(n, k, random_instance=instance_prng)

        instance = LTFArray(
            weight_array=weight_array,
            transform=transformation,
            combiner=combiner,
        )

        challenge_seed = 0xAB17D

        training_set_1 = TrainingSet(instance=instance, N=N, random_instance=RandomState(challenge_seed))
        training_set_2 = TrainingSet(instance=instance, N=N, random_instance=RandomState(challenge_seed))

        self.assertTrue(
            array_equal(training_set_1.challenges, training_set_2.challenges),
            'The challenges are not equal.',
        )


class TestInputFunctions(unittest.TestCase):
    """This Class tests the different challenge calculating functions."""

    def test_random_input(self):
        """This checks the shape, type  and values of the returned array."""
        n = 8
        rand_arr = random_input(n, random_instance=RandomState(0x5A11B))
        self.assertEqual(len(rand_arr), n, 'Array must be of length n.')
        self.assertEqual(rand_arr.dtype, dtype(RESULT_TYPE), 'Array must be of type {0}'.format(RESULT_TYPE))

        array_sum = sum(rand_arr)
        # Checks for different values
        self.assertNotEqual(array_sum, n, 'All values are 1.')
        self.assertNotEqual(array_sum, 0, 'All values are 0.')

    def test_all_inputs(self):
        """This checks the shape and type of the returned multidimensional array."""
        n = 8
        N = 2 ** n
        arr = all_inputs(n)
        self.check_multi_dimensional_array(arr, N, n, RESULT_TYPE)

    def test_random_inputs(self):
        """This checks the shape and type of the returned multidimensional array."""
        n = 8
        N = 2 ** (int(n / 2))
        rand_arrays = random_inputs(n, N)
        self.check_multi_dimensional_array(rand_arrays, N, n, RESULT_TYPE)

    def test_sample_inputs(self):
        """This checks the shape and type of the returned multidimensional array."""
        n = 8
        N = 2 ** n
        all_arr = sample_inputs(n, N)
        self.check_multi_dimensional_array(all_arr, N, n, RESULT_TYPE)

        N = 2 ** int(n / 2)
        rand_arr = sample_inputs(n, N)
        self.check_multi_dimensional_array(rand_arr, N, n, RESULT_TYPE)

    def test_transform_challenge_01_to_11(self):
        """This function checks the correct behavior of the challenge transformation."""
        challenge_01 = array([0, 1, 0, 1, 1, 0, 0, 0], dtype=RESULT_TYPE)
        challenge_11 = array([1, -1, 1, -1, -1, 1, 1, 1], dtype=RESULT_TYPE)
        transformed_challenge = transform_challenge_01_to_11(challenge_01)
        assert_array_equal(challenge_11, transformed_challenge)
        self.assertEqual(transformed_challenge.dtype, dtype(RESULT_TYPE),
                         'The array is not of type {0}.'.format(RESULT_TYPE))

    def test_transform_challenge_11_to_01(self):
        """This function checks the correct behavior of the challenge transformation."""
        challenge_11 = array([-1, 1, 1, 1, -1, 1, -1, 1], dtype=RESULT_TYPE)
        challenge_01 = array([1, 0, 0, 0, 1, 0, 1, 0], dtype=RESULT_TYPE)
        transformed_challenge = transform_challenge_11_to_01(challenge_11)
        assert_array_equal(challenge_01, transformed_challenge)
        self.assertEqual(transformed_challenge.dtype, dtype(RESULT_TYPE),
                         'The array is not of type {0}.'.format(RESULT_TYPE))

    def test_chi_vectorized(self):
        """This function checks the return type of this function."""
        n = 8
        N = 2 ** int(n / 2)
        s = random_input(n)
        inputs = random_inputs(n, N)
        chi_arr = chi_vectorized(s, inputs)
        self.assertEqual(len(chi_arr), N, 'The array must contain {0} arrays.'.format(N))
        self.assertEqual(chi_arr.dtype, RESULT_TYPE, 'The array must be of type {0}'.format(RESULT_TYPE))

    def test_poly_mult_div(self):
        """This method checks the shape and type of two dimensional arrays."""
        n = 8
        k = 2
        N = 2 ** int(n / 2)
        challenges_11 = random_inputs(n, N)
        challenges_01 = array([transform_challenge_11_to_01(c) for c in challenges_11], dtype=RESULT_TYPE)
        irreducible_polynomial = array([1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=RESULT_TYPE)
        poly_mult_div(challenges_01, irreducible_polynomial, k)
        self.check_multi_dimensional_array(challenges_01, N, n, RESULT_TYPE)

    def check_multi_dimensional_array(self, arr, arr_size, sub_arr_size, arr_type):
        """This method checks the shape and type of two dimensional arrays.
        :param arr: array of type arr_type
        :param arr_size: int
        :param sub_arr_size: int
        :param arr_type: string of numpy.dtype
        """
        self.assertEqual(len(arr), arr_size, 'The array must contain {0} arrays.'.format(arr_size))
        for i in range(arr_size):
            self.assertEqual(len(arr[i]), sub_arr_size,
                             'The sub array does not match the length of {0}.'.format(sub_arr_size))
            self.assertEqual(arr.dtype, arr_type, 'The array must be of type {0}'.format(arr_type))
