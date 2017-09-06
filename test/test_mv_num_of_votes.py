import unittest
import mv_num_of_votes


class TestMvNumOfVotes(unittest.TestCase):

    def test_8_1_puf(self):
        overall_desired_stability = 0.8
        mv_num_of_votes.main(["0.95", str(overall_desired_stability), "8", "1", "1", "0.33", "250", "1"])

        # Check if the number of results is correct
        log_file = open('exp1.0xc0deba5e_0_8_1_250_transform_id_combiner_xor.log', 'r')
        line = log_file.readline()
        while line != '':
            stability = float(line.split('\t')[0])
            if stability >= overall_desired_stability:
                break

        # if the line is '' then no stability where found which satisfy overall_desired_stability
        self.assertNotEquals(line, '')
