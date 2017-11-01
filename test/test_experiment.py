"""This module tests the different experiment classes."""
import unittest
from test.utility import remove_test_logs, logging, get_functions_with_prefix, LOG_PATH
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experiment.majority_vote import ExperimentMajorityVoteFindVotes


class TestBase(unittest.TestCase):
    """
    Every experiment needs logs in order to work. This class is used to delete all logs before after an experiment
    test."
    """

    def setUp(self):
        # Remove all log files
        remove_test_logs()

    def tearDown(self):
        # Remove all log files
        remove_test_logs()


class TestExperimentLogisticRegression(TestBase):
    """
    This class tests the logistic regression experiment.
    """
    @logging
    def test_run_and_analyze(self, logger):
        """
        This method only runs the experiment.
        """
        lr16_4 = ExperimentLogisticRegression(
            LOG_PATH+'exp1', 8, 2, 2 ** 8, 0xbeef, 0xbeef, LTFArray.transform_id,
            LTFArray.combiner_xor,
        )
        lr16_4.execute(logger.queue, logger.logger_name)


class TestExperimentMajorityVoteFindVotes(TestBase):
    """
    This class is used to test the Experiment which searches for a number of votes which is needed to achieve an
    overall desired stability.
    """
    @logging
    def test_run_and_analyze(self, logger):
        """
        This method run the experiment and checks if a number of votes was found in oder to satisfy an
        overall desired stability.
        """
        n = 8
        experiment = ExperimentMajorityVoteFindVotes(
            log_name=logger.logger_name,
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
            seed_challenges=0xf000,
            desired_stability=0.95,
            overall_desired_stability=0.8,
            minimum_vote_count=1,
            iterations=2,
            bias=None
        )
        experiment.execute(logger.queue, logger.logger_name)

        self.assertGreaterEqual(experiment.result_overall_stab, experiment.overall_desired_stability,
                                'No vote_count was found.')

    @logging
    def test_run_and_analyze_bias_list(self, logger):
        """
        This method runs the experiment with a bias list and checks if a number of votes was found in order to satisfy
        an overall desired stability.
        """
        n = 8
        experiment = ExperimentMajorityVoteFindVotes(
            log_name=logger.logger_name,
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
            seed_challenges=0xf000,
            desired_stability=0.95,
            overall_desired_stability=0.8,
            minimum_vote_count=1,
            iterations=2,
            bias=[0.001, 0.002]
        )

        experiment.execute(logger.queue, logger.logger_name)

        self.assertGreaterEqual(experiment.result_overall_stab, experiment.overall_desired_stability,
                                'No vote_count was found.')

    @logging
    def test_run_and_analyze_bias_value(self, logger):
        """
        This method runs the experiment with a bias value and checks if a number of votes was found in order to
        satisfy an overall desired stability.
        """
        n = 8
        experiment = ExperimentMajorityVoteFindVotes(
            log_name=logger.logger_name,
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
            seed_challenges=0xf000,
            desired_stability=0.95,
            overall_desired_stability=0.8,
            minimum_vote_count=1,
            iterations=2,
            bias=0.56
        )

        experiment.execute(logger.queue, logger.logger_name)

        self.assertGreaterEqual(experiment.result_overall_stab, experiment.overall_desired_stability,
                                'No vote_count was found.')
