"""This module tests the different experiment classes."""
import unittest
from multiprocessing import Queue, Process

from numpy import around
from numpy import shape
from numpy.random import RandomState
from numpy.testing import assert_array_equal

from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression, Parameters as LRParameters
from pypuf.experiments.experiment.majority_vote import ExperimentMajorityVoteFindVotes, Parameters as MVParameters
from pypuf.experiments.experiment.property_test import ExperimentPropertyTest, Parameters as PTParameters
from pypuf.experiments.experiment.reliability_based_cmaes import ExperimentReliabilityBasedCMAES
from pypuf.experiments.experimenter import log_listener, setup_logger
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from test.utility import remove_test_logs, logging, get_functions_with_prefix, LOG_PATH


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
            LOG_PATH + 'exp1',
            LRParameters(
                n=8, k=2, N=2 ** 8, seed_model=0xbeef, seed_distance=0xbeef,
                seed_instance=0xdead, seed_challenge=0xdead,
                transformation='id', combiner='xor',
                mini_batch_size=2, shuffle=False, convergence_decimals=2
            )
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
            # Check the results to be equal
            experiment_1_result = experiment_1.result._asdict()
            experiment_2_result = experiment_2.result._asdict()
            for field in [
                    'experiment_id',
                    'measured_time',
                    'model',
                    'memory_rss_max',
            ]:
                experiment_1_result.pop(field)
                experiment_2_result.pop(field)
            self.assertDictEqual(experiment_1_result, experiment_2_result)

        def get_exp(name, trans, comb):
            """Experiment creation shortcut
            :param name: string
                         Name of the experiment
            """
            return ExperimentLogisticRegression(
                LOG_PATH + name,
                LRParameters(
                    n=n, k=k, N=N, seed_model=seed_model, seed_distance=seed_distance,
                    seed_instance=seed_instance, seed_challenge=seed_challenge,
                    transformation=trans, combiner=comb,
                    mini_batch_size=0, shuffle=False, convergence_decimals=1
                )
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
            LRParameters(
                n=n, k=k, N=N, seed_model=seed_model, seed_distance=seed_distance,
                seed_instance=seed_instance, seed_challenge=seed_challenge,
                transformation='soelter_lightweight_secure', combiner='xor',
                mini_batch_size=0, shuffle=False, convergence_decimals=2
            )
        )
        experiment.execute(logger.queue, logger.logger_name)

        result_str = logger.read_result_log()
        self.assertFalse(result_str == '', 'The result log was empty.')

        error = 'LR learning results deviate from legacy learning results.'
        self.assertEqual(experiment.result.iteration_count, 274, error)
        self.assertEqual(experiment.result.epoch_count, 274, error)
        self.assertEqual(experiment.result.gradient_step_count, 274, error)
        self.assertEqual(experiment.result.accuracy, 0.98828125, error)
        assert_array_equal(
            around(experiment.result.model, decimals=8),
            around([
                0.00351544, -0.00504143, 0.01470355, 0.00481524, 0.00649157, -0.00301955,
                -0.0025765, 0.00841928, 0., 0.03248558, 0.37524702, -0.0683109,
                0.40447738, 0.49995907, 0.52796785, 0.00060493, -0.40093716, 0.,
            ], decimals=8),
            error
        )


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
            parameters=MVParameters(
                n=n,
                k=2,
                challenge_count=2 ** 8,
                seed_instance=0xC0DEBA5E,
                seed_instance_noise=0xdeadbeef,
                transformation='id',
                combiner='xor',
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
        )
        experiment.execute(logger.queue, logger.logger_name)

        self.assertGreaterEqual(experiment.result.overall_stab, experiment.parameters.overall_desired_stability,
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
            parameters=MVParameters(
                n=n,
                k=2,
                challenge_count=2 ** 8,
                seed_instance=0xC0DEBA5E,
                seed_instance_noise=0xdeadbeef,
                transformation='id',
                combiner='xor',
                mu=0,
                sigma=1,
                sigma_noise_ratio=NoisyLTFArray.sigma_noise_from_random_weights(n, 1, .5),
                seed_challenges=0xf000,
                desired_stability=0.95,
                overall_desired_stability=0.8,
                minimum_vote_count=1,
                iterations=2,
                bias=[0.001, 0.002],
            )
        )

        experiment.execute(logger.queue, logger.logger_name)

        self.assertGreaterEqual(experiment.result.overall_stab, experiment.parameters.overall_desired_stability,
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
            parameters=MVParameters(
                n=n,
                k=2,
                challenge_count=2 ** 8,
                seed_instance=0xC0DEBA5E,
                seed_instance_noise=0xdeadbeef,
                transformation='id',
                combiner='xor',
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
        )

        experiment.execute(logger.queue, logger.logger_name)

        self.assertGreaterEqual(experiment.result.overall_stab, experiment.parameters.overall_desired_stability,
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
                parameters=PTParameters(
                    test_function=test_function,
                    challenge_count=N,
                    measurements=measurements,
                    challenge_seed=challenge_seed,
                    ins_gen_function=ins_gen_function,
                    param_ins_gen=param_ins_gen,
                )
            )
        tests = ['reliability_statistic', 'uniqueness_statistic']
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
                                        'create_noisy_ltf_arrays', array_parameter)
            exp_rel.execute(logger.queue, logger.logger_name)


class TestExperimentReliabilityBasedCMAES(TestBase):
    """This class tests the reliability based CMAES experiment."""
    def test_run_and_analyze(self):
        """This method only runs the experiment."""
        logger_name = 'log'

        # Setup multiprocessing logging
        queue = Queue(-1)
        listener = Process(target=log_listener, args=(queue, setup_logger, logger_name,))
        listener.start()

        experiment = ExperimentReliabilityBasedCMAES(
            log_name=logger_name,
            seed_instance=0xbee,
            k=2,
            n=16,
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            noisiness=0.1,
            seed_challenges=0xbee,
            num=2**12,
            reps=4,
            seed_model=0xbee,
            pop_size=16,
            limit_stag=100,
            limit_iter=1000,
        )
        experiment.execute(queue, logger_name)

        queue.put_nowait(None)
        listener.join()

    def test_calc_individual_accs(self):
        """This method tests the calculation of individual (non-polarized) accuracies of a learned model."""
        exp = ExperimentReliabilityBasedCMAES(
            log_name='exp_log',
            seed_instance=0x123,
            k=2,
            n=16,
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            noisiness=.1,
            seed_challenges=0x456,
            num=40000,
            reps=5,
            seed_model=0x789,
            pop_size=24,
            limit_stag=40,
            limit_iter=500
        )
        weight_array_model = LTFArray.normal_weights(
            exp.n, exp.k, random_instance=RandomState(exp.seed_model)
        )
        exp.model = LTFArray(
            combiner=exp.combiner,
            transform=exp.transform,
            weight_array=weight_array_model
        )
        weight_array_instance = LTFArray.normal_weights(
            exp.n, exp.k, random_instance=RandomState(exp.seed_instance)
        )
        exp.instance = LTFArray(
            combiner=exp.combiner,
            transform=exp.transform,
            weight_array=weight_array_instance
        )
        individual_accs = exp.calc_individual_accs()
        self.assertIsNotNone(individual_accs)
        assert shape(individual_accs) == (exp.k,)
        for i in range(exp.k):
            assert individual_accs[i] > 0.0
            assert individual_accs[i] <= 1.0
