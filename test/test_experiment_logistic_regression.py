import unittest
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression


class TestExperimentLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.lr16_4 = ExperimentLogisticRegression('exp1.log', 8, 2, 2**8, 0xbeef, 0xbeef, LTFArray.transform_id,
                                                   LTFArray.combiner_xor, restarts=6)
        self.lr16_4.analysis()

    def test_lr_experiment_output_string(self):
        self.assertNotEqual(self.lr16_4.output_string(), '')

    def test_lr_experiment_name(self):
        self.assertNotEqual(self.lr16_4.name, '')
