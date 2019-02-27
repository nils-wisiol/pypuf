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

    def log_parameter(self, name):
        """
        This function returns a the log parameter compatible with sim_learn.
        :param name: string
                     name of the logfile
        """
        return '--log_name={0}'.format(LOG_PATH + name)

    @mute
    def test_id(self):
        """This tests the id transformation and xor combiner."""
        sim_learn.main(["8", "2", "id", "xor", "20", "1", "2", "1234", "1234", self.log_parameter("test_id")])

    @mute
    def test_atf(self):
        """This tests the atf transformation and xor combiner."""
        sim_learn.main(["8", "2", "atf", "xor", "20", "1", "2", "1234", "1234", self.log_parameter("test_atf")])

    @mute
    def test_lightweight_secure(self):
        """This tests the lightweight secure transformation and xor combiner."""
        sim_learn.main(["8", "2", "lightweight_secure", "xor", "20", "1", "2", "1234", "1234",
                        self.log_parameter("test_lightweight_secure")])

    @mute
    def test_ip_mod2_id(self):
        """This tests the identity transformation and inner product mod 2 combiner."""
        sim_learn.main(
            ["8", "2", "id", "ip_mod2", "20", "1", "2", "1234", "1234", self.log_parameter("test_ip_mod2_id")])

    @mute
    def test_ip_mod2_atf(self):
        """This tests the atf transformation and inner product mod 2 combiner."""
        sim_learn.main(
            ["8", "2", "atf", "ip_mod2", "20", "1", "2", "1234", "1234", self.log_parameter("test_ip_mod2_atf")])

    @mute
    def test_ip_mod2_lightweight_secure(self):
        """This tests the lightweight secure transformation and inner product mod 2 combiner."""
        sim_learn.main(["8", "2", "lightweight_secure", "ip_mod2", "20", "1", "2", "1234", "1234",
                        self.log_parameter("test_ip_mod2_lightweight_secure")])

    @mute
    def test_permutation_atf(self):
        """This tests the atf permutation transformation and xor combiner."""
        sim_learn.main(["8", "2", "permutation_atf", "xor", "10", "1", "2", "1234", "1234",
                        self.log_parameter("test_permutation_atf")])

    @mute
    def test_log_name(self):
        """This tests for the expected number of lines in the main log file."""
        instance_count = 2
        log_name = "test_log_name"
        sim_learn.main(
            ["8", "2", "id", "xor", "10", "1", str(instance_count), "1234", "1234", self.log_parameter(log_name)]
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
        log_file = open('logs/' + LOG_PATH + log_name + '.log', 'r')
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
        log_name = 'test_number_of_results'
        sim_learn.main(
            ["8", "2", "id", "xor", "10", str(restarts), str(instance_count), "1234", "1234",
             self.log_parameter(log_name)]
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
        log_file = open('logs/' + LOG_PATH + log_name + '.log', 'r')
        self.assertEqual(line_count(log_file), expected_number_of_result, 'Unexpected number of results')
        log_file.close()

    def read_log(self, path):
        """This function reads the content of a log file without time measuring and deletes the file.
        :param path: string
                     Path to a file
        :return: list of list of string
        """
        lines = []
        with open(path, 'r') as file:
            line = file.readline()
            while line != '':
                values = line.split(',')
                del values[0]  # Drop UUID
                del values[3]  # Drop measured time
                lines.append(values)
                line = file.readline()
        remove_test_logs(path)
        return lines

    def check_seeds(self, parameter_set1, parameter_set2, log_path):
        """
        This method checks the results of sim_learn for different parameter lists.
        :param parameter_set1: list of string
        :param parameter_set2: list of string
        :param log_path: string
        """
        sim_learn.main(parameter_set1)
        res_param_set1 = self.read_log('logs/' + log_path)

        sim_learn.main(parameter_set2)
        res_param_set2 = self.read_log('logs/' + log_path)

        # Test challenge seed impact
        # remove pid
        del res_param_set1[0][0]
        del res_param_set2[0][0]
        # remove timing info
        del res_param_set1[0][2]
        del res_param_set2[0][2]
        self.assertNotEqual(res_param_set1, res_param_set2)

        sim_learn.main(parameter_set1)
        res_param_set1_2 = self.read_log('logs/' + log_path)

        sim_learn.main(parameter_set2)
        res_param_set2_2 = self.read_log('logs/' + log_path)

        # Test challenge to be deterministic
        # remove pid
        del res_param_set1_2[0][0]
        del res_param_set2_2[0][0]
        # remove timing info
        del res_param_set1_2[0][2]
        del res_param_set2_2[0][2]
        self.assertEqual(res_param_set2, res_param_set2_2)
        self.assertEqual(res_param_set1, res_param_set1_2)

    @mute
    def test_seeds(self):
        """
        This tests the training set challenge generation and accuracy seeds.
        sim_learn must behave deterministically to pass this test.
        """
        log_name = 'test_seed_challenges'
        log_path = LOG_PATH + log_name + '.log'
        parameter = ["8", "2", "id", "xor", "20", "1", "1", "1234", "1234", self.log_parameter(log_name)]
        seed_parameter_chl = '--seed_challenges=0xBA11'
        seed_parameter_dist = '--seed_distance=5ACE'
        parameter_with_seed_chl = parameter + [seed_parameter_chl]
        parameter_with_seed_dist = parameter + [seed_parameter_dist]

        self.check_seeds(parameter, parameter_with_seed_chl, log_path)
        self.check_seeds(parameter, parameter_with_seed_dist, log_path)
        self.check_seeds(parameter_with_seed_chl, parameter_with_seed_dist, log_path)
