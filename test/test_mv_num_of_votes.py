"""
This module is used to test the command line tool which searches the number of votes for several instances of
SimulationMajorityLTFArrays in order to satisfy a overall desired stability.
"""
import unittest
from test.utility import remove_test_logs, LOG_PATH
import mv_num_of_votes


class TestMvNumOfVotes(unittest.TestCase):
    """This class is used to test the mv_num_of_votes commandline tool"""
    def setUp(self):
        # Remove all log files
        remove_test_logs()

    def tearDown(self):
        # Remove all log files
        remove_test_logs()

    def test_8_1_puf(self):
        """
        This method checks the output log of mv_num_of_votes for a stability greater equal the
        overall_desired_stability.
        """
        log_name = LOG_PATH+"test_8_1_puf"
        overall_desired_stability = 0.8
        mv_num_of_votes.main(["0.95", str(overall_desired_stability), "8", "1", "1", "0.33", "250", "1", "--log_name",
                              log_name])

        # Check if the number of results is correct
        log_file = open(log_name + '.log', 'r')
        line = log_file.readline()

        # If the line is '' then no stability where found which satisfy overall_desired_stability
        self.assertNotEqual(line, '', 'no stability where found which satisfy overall_desired_stability')
        # Get the stability entry
        stability = float(line.split('\t')[6])
        # Check the stability
        self.assertGreaterEqual(stability, overall_desired_stability)
