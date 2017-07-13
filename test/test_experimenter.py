import unittest
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experimenter import Experimenter


class TestExperimenter(unittest.TestCase):
    def test_lr_experiments(self):
        lr16_4_1 = ExperimentLogisticRegression('exp1.log', 8, 2, 2**8, 0xbeef, 0xbeef, LTFArray.transform_id,
                                                LTFArray.combiner_xor, restarts=6)
        lr16_4_2 = ExperimentLogisticRegression('exp2.log', 8, 2, 2**8, 0xbeef, 0xbeef, LTFArray.transform_id,
                                                LTFArray.combiner_xor, restarts=6)
        lr16_4_3 = ExperimentLogisticRegression('exp3.log', 8, 2, 2**8, 0xbeef, 0xbeef, LTFArray.transform_id,
                                                LTFArray.combiner_xor, restarts=6)
        lr16_4_4 = ExperimentLogisticRegression('exp4.log', 8, 2, 2**8, 0xbeef, 0xbeef, LTFArray.transform_id,
                                                LTFArray.combiner_xor, restarts=6)
        lr16_4_5 = ExperimentLogisticRegression('exp5.log', 8, 2, 2**8, 0xbeef, 0xbeef, LTFArray.transform_id,
                                                LTFArray.combiner_xor, restarts=6)
        experiments = [lr16_4_1, lr16_4_2, lr16_4_3, lr16_4_4, lr16_4_5]
        experimenter = Experimenter('log', experiments)
        experimenter.run()
