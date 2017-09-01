import unittest
import os
import glob
import multiprocessing
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experimenter import log_listener, setup_logger
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experiment.reliability_based_cma_es import ExperimentReliabilityBasedCMAES


class TestExperimentLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Remove all log files
        paths = list(glob.glob('*.log'))
        for p in paths:
            os.remove(p)

    def tearDown(self):
        # Remove all log files
        paths = list(glob.glob('*.log'))
        for p in paths:
            os.remove(p)

    def test_run_and_analyze(self):
        logger_name = 'log'

        # Setup multiprocessing logging
        queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(target=log_listener,
                                           args=(queue, setup_logger, logger_name,))
        listener.start()

        self.lr16_4 = ExperimentLogisticRegression('exp1', 8, 2, 2**8, 0xbeef, 0xbeef, LTFArray.transform_id,
                                                   LTFArray.combiner_xor)
        self.lr16_4.execute(queue, logger_name)

        queue.put_nowait(None)
        listener.join()


class TestExperimentReliabilityBasedCMAES(unittest.TestCase):
    def setUp(self):
        # Remove all log files
        paths = list(glob.glob('*.log'))
        for p in paths:
            os.remove(p)

    def tearDown(self):
        # Remove all log files
        paths = list(glob.glob('*.log'))
        for p in paths:
            os.remove(p)

    def test_run_and_analyze(self):
        logger_name = 'log'

        # Setup multiprocessing logging
        queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(target=log_listener,
                                           args=(queue, setup_logger, logger_name,))
        listener.start()

        self.exp = ExperimentReliabilityBasedCMAES('exp1', 8, 1, 2**8-1, 0xDEADBEEF, 0xBAADF00D, 0xDEADDA7A,
                                                   LTFArray.transform_id, LTFArray.combiner_xor, 0, 1, 1, 5, 10,
                                                   1 / 2 ** 8, 1024, 0xABAD1DEA, bias=False)
        self.exp.execute(queue, logger_name)

        queue.put_nowait(None)
        listener.join()
