"""This module tests the different experiment classes."""
import unittest
from collections import OrderedDict
from test.utility import remove_test_logs, logging, get_functions_with_prefix, LOG_PATH
from numpy import pi, sqrt
from numpy.random import RandomState
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.simulation.fourier_based.bent import BentFunctionIpMod2
from pypuf.simulation.fourier_based.dictator import Dictator
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experiment.majority_vote import ExperimentMajorityVoteFindVotes
from pypuf.experiments.experiment.fourier_coefficient import ExperimentFCCRP, ExperimentCFCA
from pypuf.property_test.base import PropertyTest
from pypuf.experiments.experiment.property_test import ExperimentPropertyTest
from pypuf.tools import sample_inputs


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
            self.assertFalse(result_1 == '', 'The experiment {0} log was empty.'.format(experiment_1.log_name))
            self.assertFalse(result_2 == '', 'The experiment log {0} was empty.'.format(experiment_2.log_name))
            # Compare logs
            self.assertTrue(result_1 == result_2,
                            'The results of {0} and {1} must be equal.'.format(experiment_1.log_name,
                                                                               experiment_2.log_name))

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
                log_name=logger.logger_name,
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
            with open(exp_rel.log_name+'.log', 'r') as log_file:
                self.assertNotEqual(log_file.read(), '')


class TestExperimentFCCRP(TestBase):
    """This class test the Fourier coefficient approximation experiment."""

    @logging
    def test_experiment_fccrp(self, logger):
        """This function just executes the Fourier coefficient approximation experiment."""
        challenge_count = 100000
        instance_parameter = OrderedDict()
        instance_parameter['n'] = 8
        instance_parameter['k'] = 1
        instance_parameter['weight_random_seed'] = int(pi*1000)+0xbeef
        exp = ExperimentFCCRP(
            log_name=logger.logger_name,
            challenge_count=challenge_count,
            challenge_seed=int(pi*100)+1234,
            instance_gen=ExperimentPropertyTest.create_ltf_arrays,
            instance_parameter=instance_parameter
        )
        exp.execute(logger.queue, logger_name=logger.logger_name)


class TestExperimentExperimentCFCA(TestBase):
    """This class test the cumulative degree one weight approximation."""

    def test_degree_one_approximation(self):
        """This method tests the degree one weight approximation."""
        n = 8
        random_instance = RandomState(0x5123)

        bent_instance = BentFunctionIpMod2(n)
        dictator_instance = Dictator(4, n)
        challenges = sample_inputs(n, 2**n, random_instance=random_instance)
        responses = bent_instance.eval(challenges)
        degree_on_weights = ExperimentCFCA.approx_degree_one_weight(responses, challenges, 1, 2**n)
        # Check the bent function approximation
        self.assertEqual(round(n*1/sqrt(2**n)**2, 2), round(degree_on_weights[-1], 2))

        responses = dictator_instance.eval(challenges)
        degree_on_weights = ExperimentCFCA.approx_degree_one_weight(responses, challenges, 1, 2**n)
        # Check the dictator approximation
        self.assertEqual(1.0, round(degree_on_weights[-1], 1))

        ltfarray_instance = LTFArray(
            weight_array=LTFArray.normal_weights(n, 1, random_instance=random_instance),
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor
        )
        responses = ltfarray_instance.eval(challenges)
        degree_on_weights = ExperimentCFCA.approx_degree_one_weight(responses, challenges, 1, 2**n)
        # Check a fix result
        self.assertEqual(0.66928, round(degree_on_weights[-1], 5))
