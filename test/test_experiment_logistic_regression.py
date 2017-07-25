import unittest
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression


class TestExperimentLogisticRegression(unittest.TestCase):
    def test_run_and_analyze(self):
        self.lr16_4 = ExperimentLogisticRegression('exp1.log', 8, 2, 2**8, 0xbeef, 0xbeef, LTFArray.transform_id,
                                                   LTFArray.combiner_xor)
        self.lr16_4.execute()
