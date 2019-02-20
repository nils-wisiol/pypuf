"""This module tests the different experiment classes."""
import unittest
from test.utility import remove_test_logs, logging, get_functions_with_prefix, LOG_PATH
from numpy import array
from numpy.testing import assert_array_equal
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experiment.majority_vote import ExperimentMajorityVoteFindVotes
from pypuf.property_test.base import PropertyTest
from pypuf.experiments.experiment.property_test import ExperimentPropertyTest


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
            LOG_PATH + 'exp1', 8, 2, 2 ** 8, 0xbeef, 0xbeef, LTFArray.transform_id,
            LTFArray.combiner_xor,
        )
        lr16_4.execute(logger.queue, logger.logger_name)

    @logging
    def test_fix_result(self, logger):
        """
        This test the experiment to have a deterministic result.
        """
        n = 64
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
            exp_1_result_log = open('logs/' + experiment_1.progress_log_name + '.log', 'r')
            exp_2_result_log = open('logs/' + experiment_2.progress_log_name + '.log', 'r')
            # Save the results
            result_1 = exp_1_result_log.read()
            result_2 = exp_2_result_log.read()
            # Close logs
            exp_1_result_log.close()
            exp_2_result_log.close()
            # Check the results to be not empty
            self.assertFalse(result_1 == '', 'The experiment log {0} was empty.'.format(experiment_1.progress_log_name))
            self.assertFalse(result_2 == '', 'The experiment log {0} was empty.'.format(experiment_2.progress_log_name))
            # Compare logs
            self.assertTrue(result_1 == result_2,
                            'The results of {0} and {1} must be equal.'.format(experiment_1.progress_log_name,
                                                                               experiment_2.progress_log_name))

        def get_exp(name, trans, comb):
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
                experiment_1 = get_exp('exp1', transformation, combiner)
                experiment_2 = get_exp('exp2', transformation, combiner)
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

        legacy_result = ['0xbae55e', '0x5c6ae1e', 'None', '8', '2', '255',
                         'transform_soelter_lightweight_secure',
                         'combiner_xor', '256', '256', '2.000000', '0.988281',
                         '0.003990716152,-0.005655328619,0.016386240611,0.005377622618,0.007297814222,-0.003351419305,'
                         '-0.002956429735,0.009401146144,0.000000126573,0.034918353082,0.368758330023,-0.078502828629,'
                         '0.417595993772,0.509973673286,0.513855115932,0.000297216086,-0.396978991707,-0.005413902281'
                         '\n']
        result_str = logger.read_result_log()
        self.assertFalse(result_str == '', 'The result log was empty.')
        experiment_result = result_str.split('\t')
        # remove execution time
        del experiment_result[11]
        assert_array_equal(experiment_result, legacy_result, 'You changed the Logistic Regression Learner'
                                                             'significantly.')

    def test_mathematica_compatibility(self):
        """
        Tests if the result log of Logistic Regression learning is compatible with Mathematica input, i.e. it must not
        contain numbers in scientific notation.
        """
        experiment = ExperimentLogisticRegression(
            LOG_PATH + 'exp',
            n=2,
            k=1,
            N=1,
            seed_instance=1,
            seed_model=2,
            transformation=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
        )
        experiment.execute(None, 'testlog')
        experiment.model = LTFArray(
            weight_array=array([[1, 10E-13]]),
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
        )
        with self.assertLogs('testlog', level='DEBUG') as mock_logger:
            experiment.analyze()
        self.assertEqual(mock_logger.output[0].split('\t')[13], '1.000000000000,0.000000000001,0.000000000000')


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
            progress_log_prefix=logger.logger_name,
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
            progress_log_prefix=logger.logger_name,
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
            progress_log_prefix=logger.logger_name,
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


class TestExperimentPropertyTest(TestBase):
    """This class is used to test the property test experiment"""

    @logging
    def test_run_and_analyze_tests(self, logger):
        """
        This method runs the property testing experiment with different parameters.
        """

        def create_experiment(N, test_function, ins_gen_function, param_ins_gen):
            """A shortcut function to create property experiments"""
            measurements = 10
            challenge_seed = 0xDE5
            return ExperimentPropertyTest(
                progress_log_name=logger.logger_name,
                test_function=test_function,
                challenge_count=N,
                measurements=measurements,
                challenge_seed=challenge_seed,
                ins_gen_function=ins_gen_function,
                param_ins_gen=param_ins_gen,
            )
        tests = [PropertyTest.reliability_statistic, PropertyTest.uniqueness_statistic]
        N = 255
        array_parameter = {
            'n': 16,
            'k': 1,
            'instance_count': 10,
            'transformation': NoisyLTFArray.transform_id,
            'combiner': NoisyLTFArray.combiner_xor,
            'bias': None,
            'mu': 0,
            'sigma': 1,
            'weight_random_seed': 0xF4EE,
            'sigma_noise': 0.5,
            'noise_random_seed': 0xEE4F,
        }
        for test_function in tests:
            exp_rel = create_experiment(N, test_function,
                                        ExperimentPropertyTest.create_noisy_ltf_arrays, array_parameter)
            exp_rel.execute(logger.queue, logger.logger_name)
            with open('logs/' + exp_rel.progress_log_name + '.log', 'r') as log_file:
                self.assertNotEqual(log_file.read(), '')
