"""This module test the experimenter class which is used to distribute experiments over several cores."""
import unittest
import os
import glob
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.experiments.experiment.base import Experiment
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experiment.majority_vote import ExperimentMajorityVoteFindVotes
from pypuf.experiments.experimenter import Experimenter


class TestExperimenter(unittest.TestCase):
    """
    This class is used to test the experimenter class.
    """
    def setUp(self):
        # Remove all log files
        paths = list(glob.glob('*.log'))
        for path in paths:
            os.remove(path)

    def tearDown(self):
        # Remove all log files
        paths = list(glob.glob('*.log'))
        for path in paths:
            os.remove(path)

    def test_lr_experiments(self):
        """This method runs the experimenter for four logistic regression experiments."""
        lr16_4_1 = ExperimentLogisticRegression('test_lr_experiments1', 8, 2, 2 ** 8, 0xbeef, 0xbeef,
                                                LTFArray.transform_id,
                                                LTFArray.combiner_xor)
        lr16_4_2 = ExperimentLogisticRegression('test_lr_experiments2', 8, 2, 2 ** 8, 0xbeef, 0xbeef,
                                                LTFArray.transform_id,
                                                LTFArray.combiner_xor)
        lr16_4_3 = ExperimentLogisticRegression('test_lr_experiments3', 8, 2, 2 ** 8, 0xbeef, 0xbeef,
                                                LTFArray.transform_id,
                                                LTFArray.combiner_xor)
        lr16_4_4 = ExperimentLogisticRegression('test_lr_experiments4', 8, 2, 2 ** 8, 0xbeef, 0xbeef,
                                                LTFArray.transform_id,
                                                LTFArray.combiner_xor)
        lr16_4_5 = ExperimentLogisticRegression('test_lr_experiments5', 8, 2, 2 ** 8, 0xbeef, 0xbeef,
                                                LTFArray.transform_id,
                                                LTFArray.combiner_xor)
        experiments = [lr16_4_1, lr16_4_2, lr16_4_3, lr16_4_4, lr16_4_5]
        experimenter = Experimenter('log', experiments)
        experimenter.run()

    def test_mv_experiments(self):
        """This method runs the experimenter with five ExperimentMajorityVoteFindVotes experiments."""
        experiments = []
        for i in range(5):
            n = 8
            logger_name = 'test_mv_exp{0}'.format(i)
            experiment = ExperimentMajorityVoteFindVotes(
                log_name=logger_name,
                n=n,
                k=2,
                challenge_count=2 ** 8,
                seed_instance=0xC0DEBA5E,
                seed_instance_noise=0xdeadbeef,
                transformation=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
                mu=0,
                sigma=1,
                sigma_noise_ratio=NoisyLTFArray.sigma_noise_from_random_weights(n, 1, .5),
                seed_challenges=0xf000+i,
                desired_stability=0.95,
                overall_desired_stability=0.8,
                minimum_vote_count=1,
                iterations=2,
                bias=None
            )
            experiments.append(experiment)
        experimenter = Experimenter('test_mv_experimenter', experiments)
        experimenter.run()

    def test_multiprocessing_logs(self):
        """
        This test checks for the predicted amount for result.
        """
        experiments = []
        n = 28
        for i in range(n):
            log_name = 'test_multiprocessing_logs{0}'.format(i)
            lr16_4_1 = ExperimentLogisticRegression(log_name, 8, 2, 2 ** 8, 0xbeef, 0xbeef,
                                                    LTFArray.transform_id,
                                                    LTFArray.combiner_xor)
            experiments.append(lr16_4_1)

        for i in range(n):
            log_name = 'test_multiprocessing_logs{0}'.format(i)
            experiment = ExperimentMajorityVoteFindVotes(
                log_name=log_name,
                n=8,
                k=2,
                challenge_count=2 ** 8,
                seed_instance=0xC0DEBA5E,
                seed_instance_noise=0xdeadbeef,
                transformation=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
                mu=0,
                sigma=1,
                sigma_noise_ratio=NoisyLTFArray.sigma_noise_from_random_weights(n, 1, .5),
                seed_challenges=0xf000 + i,
                desired_stability=0.95,
                overall_desired_stability=0.8,
                minimum_vote_count=1,
                iterations=2,
                bias=None
            )
            experiments.append(experiment)

        experimenter = Experimenter('test_multiprocessing_logs', experiments)
        experimenter.run()

        def line_count(file_object):
            """
            :param file_object:
            :return: number of lines
            """
            count = 0
            while file_object.readline() != '':
                count = count + 1
            return count

        paths = list(glob.glob('*.log'))
        # Check if the number of lines is greater than zero
        for log_path in paths:
            exp_log_file = open(log_path, 'r')
            self.assertGreater(line_count(exp_log_file), 0, 'The experiment log is empty.')
            exp_log_file.close()

        # Check if the number of results is correct
        log_file = open('test_multiprocessing_logs.log', 'r')
        self.assertEqual(line_count(log_file), n*2, 'Unexpected number of results')
        log_file.close()

    def test_file_handle(self):
        """
        This test check if process file handles are deleted. Some Systems have have limit of open file handles.
        """
        class ExperimentDummy(Experiment):
            """
            This is an empty experiment class which can be used to run a huge amount of experiments with an
            experimenter.
            """
            def run(self):
                pass

            def analyze(self):
                pass

        experiments = []
        n = 1024
        for i in range(n):
            log_name = 'fail{0}'.format(i)
            experiments.append(ExperimentDummy(log_name))

        experimenter = Experimenter('fail', experiments)
        experimenter.run()
