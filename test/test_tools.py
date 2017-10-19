"""This module is used to test the functions which are implemented in pypuf.tools."""
import unittest
from numpy import zeros, dtype, array_equal
from pypuf.tools import append_last


class TestAppendLast(unittest.TestCase):
    """This class is used to test the append_last method from tools."""
    def test_same_dtype(self):
        """This function checks for type safety"""
        typ = dtype('int64')
        arr = zeros((3, 2, 1), dtype=typ)
        item = 30
        arr_res = append_last(arr, item)
        self.assertEqual(arr_res.dtype, typ, 'The types must be equal.')

        arr = zeros((3, 2, 1), dtype=typ)
        item = 30.1
        with self.assertRaisesRegex(AssertionError, 'The elements of arr and item must be of the same type.'):
            append_last(arr, item)

    def test_proper_dimension(self):
        """This function checks if the function recognize that the array have no dimensions."""
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
            shape_res[-1] = shape_res[-1]+1
            # generate the arrays
            arr_check = zeros(shape_res, dtype=typ)
            arr = zeros(shape, dtype=typ)
            # append the last item
            arr_res = append_last(arr, item)
            # check for equality
            self.assertTrue(array_equal(arr_res, arr_check), 'The arrays should be equal.')
