import unittest
import example


class TestExample(unittest.TestCase):
    def test_default(self):
        example.main()
