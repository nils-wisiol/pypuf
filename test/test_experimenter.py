"""This module test the experimenter class which is used to distribute experiments over several cores."""
import unittest
import glob
from test.utility import remove_test_logs, LOG_PATH, mute
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.experiments.experiment.base import Experiment
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experiment.majority_vote import ExperimentMajorityVoteFindVotes
from pypuf.experiments.experimenter import Experimenter, FailedExperimentsException


class TestExperimenter(unittest.TestCase):
    """
    This class is used to test the experimenter class.
    """
    def setUp(self):
        # Remove all log files
        remove_test_logs()

    def tearDown(self):
        # Remove all log files
        remove_test_logs()

    @mute
    def test_lr_experiments(self):
        """This method runs the experimenter for five logistic regression experiments."""
        experimenter = Experimenter(LOG_PATH+'test_lr_experiments')
        for i in range(5):
            experimenter.queue(
                ExperimentLogisticRegression(LOG_PATH + 'test_lr_experiments{}'.format(i+1),
                                             8, 2, 2 ** 8, 0xbeef, 0xbeef,
                                             LTFArray.transform_id,
                                             LTFArray.combiner_xor)
            )
        experimenter.run()

    @mute
    def test_mv_experiments(self):
        """This method runs the experimenter with five ExperimentMajorityVoteFindVotes experiments."""
        experimenter = Experimenter(LOG_PATH+'test_mv_experiments')
        for i in range(5):
            n = 8
            logger_name = LOG_PATH+'test_mv_exp{0}'.format(i)
            experiment = ExperimentMajorityVoteFindVotes(
                progress_log_prefix=logger_name,
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
            experimenter.queue(experiment)
        experimenter.run()

    @mute
    def test_multiprocessing_logs(self):
        """
        This test checks for the predicted amount for result.
        """
        experimenter_log_name = LOG_PATH+'test_multiprocessing_logs'
        experimenter = Experimenter(experimenter_log_name)

        n = 4
        for i in range(n):
            log_name = LOG_PATH+'test_multiprocessing_logs{0}'.format(i)
            experimenter.queue(
                ExperimentLogisticRegression(
                    log_name, 8, 2, 2 ** 8, 0xbeef, 0xbeef,
                    LTFArray.transform_id,
                    LTFArray.combiner_xor
                )
            )

        for i in range(n):
            log_name = LOG_PATH+'test_multiprocessing_logs{0}'.format(i)
            experiment = ExperimentMajorityVoteFindVotes(
                progress_log_prefix=log_name,
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
                overall_desired_stability=0.6,
                minimum_vote_count=1,
                iterations=2,
                bias=None
            )
            experimenter.queue(experiment)

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

        paths = list(glob.glob('logs/' + LOG_PATH + '*.log'))
        # Check if the number of lines is greater than zero
        for log_path in paths:
            exp_log_file = open(log_path, 'r')
            self.assertGreater(line_count(exp_log_file), 0, 'The experiment log {} is empty.'.format(log_path))
            exp_log_file.close()

        # Check if the number of results is correct
        log_file = open('logs/' + experimenter_log_name + '.log', 'r')
        self.assertEqual(line_count(log_file), n*2, 'Unexpected number of results')
        log_file.close()

    @mute
    def test_broken_experiment(self):
        """
        Verify the experimenter handles experiments that raise exceptions correctly.
        """
        experimenter = Experimenter(LOG_PATH + 'test_broken_experiments')
        experimenter.queue(ExperimentBroken('foobar'))
        experimenter.queue(ExperimentBroken('foobaz'))
        with self.assertRaises(FailedExperimentsException):
            experimenter.run()


class ExperimentDummy(Experiment):
    """
    This is an empty experiment class which can be used to run a huge amount of experiments with an
    experimenter.
    """
    def run(self):
        pass

    def analyze(self):
        pass


class ExperimentBroken(ExperimentDummy):
    """
    This experiment always raises an exception.
    """

    def run(self):
        raise Exception("Intentionally broken!")
