"""This module tests the sim_learn command line script."""
import unittest
import sim_learn


class TestSimLearn(unittest.TestCase):
    """
    This class is used to test different parameters for the sim_learn module.
    """
    def test_id(self):
        """This tests the id transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "id", "xor", "20", "1", "2", "1234", "1234"])

    def test_atf(self):
        """This tests the atf transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "atf", "xor", "20", "1", "2", "1234", "1234"])

    def test_mm(self):
        """This tests the mm transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "mm", "xor", "20", "1", "2", "1234", "1234"])

    def test_lightweight_secure(self):
        """This tests the lightweight secure transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "lightweight_secure", "xor", "20", "1", "2", "1234", "1234"])

    def test_lightweight_secure_original(self):
        """This tests the lightweight secure original transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "lightweight_secure_original", "xor", "20", "1", "2", "1234", "1234"])

    def test_1_n_bent(self):
        """This tests the one to n bent transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "1_n_bent", "xor", "20", "1", "2", "1234", "1234"])

    def test_1_1_bent(self):
        """This tests the one to one bent transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "1_1_bent", "xor", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_id(self):
        """This tests the identity transformation and inner product mod 2 combiner."""
        sim_learn.main(["sim_learn", "8", "2", "id", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_atf(self):
        """This tests the atf transformation and inner product mod 2 combiner."""
        sim_learn.main(["sim_learn", "8", "2", "atf", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_mm(self):
        """This tests the mm transformation and inner product mod 2 combiner."""
        sim_learn.main(["sim_learn", "8", "2", "mm", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_lightweight_secure(self):
        """This tests the lightweight secure transformation and inner product mod 2 combiner."""
        sim_learn.main(["sim_learn", "8", "2", "lightweight_secure", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_1_n_bent(self):
        """This tests the one to n bent transformation and inner product mod 2 combiner."""
        sim_learn.main(["sim_learn", "8", "2", "1_n_bent", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_1_1_bent(self):
        """This tests the one to one bent transformation and inner product mod 2 combiner."""
        sim_learn.main(["sim_learn", "8", "2", "1_1_bent", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_permutation_atf(self):
        """This tests the atf permutation transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "permutation_atf", "xor", "10", "1", "2", "1234", "1234"])
