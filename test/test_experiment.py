"""This module tests the different experiment classes."""
import unittest
import os
import glob
import multiprocessing
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.experiments.experimenter import log_listener, setup_logger
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experiment.majority_vote import ExperimentMajorityVoteFindVotes


class TestBase(unittest.TestCase):
    """
    Every experiment needs logs in order to work. This class is used to delete all logs before after an experiment
    test."
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


class TestExperimentLogisticRegression(TestBase):
    """
    This class tests the logistic regression experiment.
    """
    def test_run_and_analyze(self):
        """
        This method only runs the experiment.
        """
        logger_name = 'log'

        # Setup multiprocessing logging
        queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(target=log_listener,
                                           args=(queue, setup_logger, logger_name,))
        listener.start()

        lr16_4 = ExperimentLogisticRegression('exp1', 8, 2, 2 ** 8, 0xbeef, 0xbeef, LTFArray.transform_id,
                                              LTFArray.combiner_xor)
        lr16_4.execute(queue, logger_name)

        queue.put_nowait(None)
        listener.join()


class TestExperimentMajorityVoteFindVotes(unittest.TestCase):
    """
    This class is used to test the Experiment which searches for a number of votes which is needed to achieve an
    overall desired stability.
    """
    def test_run_and_analyze(self):
        """
        This method run the experiment and checks if a number of votes was found in oder to satisfy an
        overall desired stability.
        """
        logger_name = 'log'

        # Setup multiprocessing logging
        queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(target=log_listener,
                                           args=(queue, setup_logger, logger_name,))
        listener.start()

        n = 8
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
            seed_challenges=0xf000,
            desired_stability=0.95,
            overall_desired_stability=0.8,
            minimum_vote_count=1,
            iterations=2,
            bias=None
        )
        experiment.execute(queue, logger_name)

        self.assertGreaterEqual(experiment.result_overall_stab, experiment.overall_desired_stability,
                                'No vote_count was found.')

        queue.put_nowait(None)
        listener.join()

    def test_run_and_analyze_bias_list(self):
        """
        This method runs the experiment with a bias list and checks if a number of votes was found in order to satisfy
        an overall desired stability.
        """
        logger_name = 'log'

        # Setup multiprocessing logging
        queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(target=log_listener,
                                           args=(queue, setup_logger, logger_name,))
        listener.start()

        n = 8
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
            seed_challenges=0xf000,
            desired_stability=0.95,
            overall_desired_stability=0.8,
            minimum_vote_count=1,
            iterations=2,
            bias=[0.001, 0.002]
        )

        experiment.execute(queue, logger_name)

        self.assertGreaterEqual(experiment.result_overall_stab, experiment.overall_desired_stability,
                                'No vote_count was found.')

        queue.put_nowait(None)
        listener.join()

    def test_run_and_analyze_bias_value(self):
        """
        This method runs the experiment with a bias value and checks if a number of votes was found in order to
        satisfy an overall desired stability.
        """
        logger_name = 'log'

        # Setup multiprocessing logging
        queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(target=log_listener,
                                           args=(queue, setup_logger, logger_name,))
        listener.start()

        n = 8
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
            seed_challenges=0xf000,
            desired_stability=0.95,
            overall_desired_stability=0.8,
            minimum_vote_count=1,
            iterations=2,
            bias=0.56
        )

        experiment.execute(queue, logger_name)

        self.assertGreaterEqual(experiment.result_overall_stab, experiment.overall_desired_stability,
                                'No vote_count was found.')

        queue.put_nowait(None)
        listener.join()
