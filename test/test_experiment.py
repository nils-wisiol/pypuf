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

    @logging
    def test_fix_result(self, logger):
        """
        This test the experiment to have a deterministic result.
        """
        n = 8
        k = 2
        N = 8
        seed_instance = 0xBAE55E
        seed_model = 0x5C6AE1E
        seed_challenge = 10
        seed_distance = 11
        combiners = get_functions_with_prefix('combiner_', LTFArray)
        transformations = get_functions_with_prefix('transform_', LTFArray)

        def check_experiments(experiment_1, experiment_2):
            """
            This function compares the results of two experiments.
            :param experiment_1: ExperimentLogisticRegression
            :param experiment_2: ExperimentLogisticRegression
            """
            # Execute experiments
            experiment_1.execute(logger.queue, logger.logger_name)
            experiment_2.execute(logger.queue, logger.logger_name)
            # Open logs
            exp_1_result_log = open(experiment_1.log_name + '.log', 'r')
            exp_2_result_log = open(experiment_2.log_name + '.log', 'r')
            # Save the results
            result_1 = exp_1_result_log.read()
            result_2 = exp_2_result_log.read()
            # Close logs
            exp_1_result_log.close()
            exp_2_result_log.close()
            # Check the results to be not empty
            self.assertFalse(result_1 == '', 'The experiment log was empty.')
            self.assertFalse(result_2 == '', 'The experiment log was empty.')
            # Compare logs
            self.assertTrue(result_1 == result_2, 'The results must be equal.')

        def get_exp(name, k, trans, comb):
            """Experiment creation shortcut
            :param name: string
                         Name of the experiment
            """
            return ExperimentLogisticRegression(
                LOG_PATH + name,
                n,
                k,
                N,
                seed_instance,
                seed_model,
                trans,
                comb,
                seed_challenge=seed_challenge,
                seed_chl_distance=seed_distance,
            )
        # Result check
        for transformation in transformations:
            for combiner in combiners:
                experiment_1 = get_exp('exp1', k, transformation, combiner)
                experiment_2 = get_exp('exp2', k, transformation, combiner)
                check_experiments(experiment_1, experiment_2)

    @logging
    def test_fix_convergence(self, logger):
        """
        This methods checks the results of a ExperimentLogisticRegression to match a recorded result which was
        generated with the same parameters.
        """
        n = 8
        k = 2
        N = 255
        seed_instance = 0xBAE55E
        seed_model = 0x5C6AE1E
        seed_challenge = 0xB0661E
        seed_distance = 0xB0C
        experiment = ExperimentLogisticRegression(
            LOG_PATH + 'exp',
            n,
            k,
            N,
            seed_instance,
            seed_model,
            LTFArray.transform_soelter_lightweight_secure,
            LTFArray.combiner_xor,
            seed_challenge=seed_challenge,
            seed_chl_distance=seed_distance,
        )
        experiment.execute(logger.queue, logger.logger_name)

        legacy_result = ['0xbae55e', '0x5c6ae1e', '0', '8', '2', '255', 'transform_soelter_lightweight_secure',
                         'combiner_xor', '363', '1.000000', '0.00443419669755,-0.00616546911566,0.0186346081194,'
                                                            '0.0061619719475,0.00795284461334,-0.00443539877583,'
                                                            '-0.00316047872599,0.00993214368373,0.0507595729459,'
                                                            '0.415207373134,-0.0517173737839,0.285900582842,'
                                                            '0.467512016377,0.550102231366,-0.000739711610042,'
                                                            '-0.467757977178\n']
        result_str = logger.read_result_log()
        self.assertFalse(result_str == '', 'The result log was empty.')
        experiment_result = result_str.split('\t')
        # remove execution time
        del experiment_result[9]
        self.assertTrue(experiment_result == legacy_result, 'You changed the code significant.')


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
