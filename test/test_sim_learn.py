"""This module tests the sim_learn command line script."""
import unittest
from test.utility import remove_test_logs, LOG_PATH, mute
import sim_learn


class TestSimLearn(unittest.TestCase):
    """
    This class is used to test different parameters for the sim_learn module.
    """
    def setUp(self):
        # Remove all log files
        remove_test_logs()

    def tearDown(self):
        # Remove all log files
        remove_test_logs()

    @mute
    def test_id(self):
        """This tests the id transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "id", "xor", "20", "1", "2", "1234", "1234", LOG_PATH + "test_id"])

    @mute
    def test_atf(self):
        """This tests the atf transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "atf", "xor", "20", "1", "2", "1234", "1234", LOG_PATH + "test_atf"])

    @mute
    def test_lightweight_secure(self):
        """This tests the lightweight secure original transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "lightweight_secure", "xor", "20", "1", "2", "1234", "1234",
                        LOG_PATH+"test_lightweight_secure_original"])

    @mute
    def test_ip_mod2_id(self):
        """This tests the identity transformation and inner product mod 2 combiner."""
        sim_learn.main(["sim_learn", "8", "2", "id", "ip_mod2", "20", "1", "2", "1234", "1234",
                        LOG_PATH + "test_ip_mod2_id"])

    @mute
    def test_ip_mod2_atf(self):
        """This tests the atf transformation and inner product mod 2 combiner."""
        sim_learn.main(["sim_learn", "8", "2", "atf", "ip_mod2", "20", "1", "2", "1234", "1234",
                        LOG_PATH + "test_ip_mod2_atf"])

    @mute
    def test_ip_mod2_lightweight_secure(self):
        """This tests the lightweight secure transformation and inner product mod 2 combiner."""
        sim_learn.main(["sim_learn", "8", "2", "lightweight_secure", "ip_mod2", "20", "1", "2", "1234", "1234",
                        LOG_PATH + "test_ip_mod2_lightweight_secure"])

    @mute
    def test_permutation_atf(self):
        """This tests the atf permutation transformation and xor combiner."""
        sim_learn.main(["sim_learn", "8", "2", "permutation_atf", "xor", "10", "1", "2", "1234", "1234",
                        LOG_PATH + "test_permutation_atf"])

    @mute
    def test_log_name(self):
        """This tests for the expected number of lines in the main log file."""
        instance_count = 2
        log_name = LOG_PATH + 'test_log_name'
        sim_learn.main(["sim_learn", "8", "2", "id", "xor", "10", "1", str(instance_count), "1234", "1234", log_name])

        def line_count(file_object):
            """
            :param file_object:
            :return: number of lines
            """
            count = 0
            while file_object.readline() != '':
                count = count + 1
            return count

        # Check if the number of results is correct
        log_file = open(log_name + '.log', 'r')
        self.assertEqual(line_count(log_file), instance_count, 'Unexpected number of results')
        log_file.close()

    @mute
    def test_number_of_results(self):
        """
        This test checks the number of results to match a previous calculated value.
        """
        instance_count = 7
        restarts = 13
        expected_number_of_result = instance_count * restarts
        log_name = LOG_PATH + 'test_number_of_results'
        sim_learn.main(
            ["sim_learn", "8", "2", "id", "xor", "10", str(restarts), str(instance_count), "1234", "1234", log_name]
        )

        def line_count(file_object):
            """
            :param file_object:
            :return: number of lines
            """
            count = 0
            while file_object.readline() != '':
                count = count + 1
            return count

        # Check if the number of results is correct
        log_file = open(log_name + '.log', 'r')
        self.assertEqual(line_count(log_file), expected_number_of_result, 'Unexpected number of results')
        log_file.close()
