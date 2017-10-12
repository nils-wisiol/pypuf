import unittest
import input_trans_comparison
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


class TestInputTransComparison(unittest.TestCase):
    def test_id(self):
        input_trans_comparison.run_input_trans_comparison(
            n=24,
            k=2,
            transformations=[
                LTFArray.transform_id,
                LTFArray.transform_polynomial,
            ],
            Ns=[
                1200,
                2000,
            ],
            instance_sample_size=2,
            initial_seed_model=0xbeef,
        )
