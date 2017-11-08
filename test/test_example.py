"""
This module ensures that the example code is running with the current version of the framework.
"""
import unittest
from test.utility import mute
import example


class TestExample(unittest.TestCase):
    """
    This class is used to execute the code examples.
    """
    @mute
    def test_default(self):
        """
        This method just runs the example code in order to test for runtime errors.
        """
        example.main()
