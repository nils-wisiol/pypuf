"""
This module ensures that the example code is running with the current version of the framework.
"""
import unittest
import example


class TestExample(unittest.TestCase):
    """
    This class is used to execute the code examples.
    """
    def test_default(self):
        """
        This method just runs the example code in order to test for runtime errors.
        """
        example.main()
