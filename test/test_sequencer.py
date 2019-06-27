import unittest

from numpy import where

from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.studies.randomness.nist import NISTRandomnessExperiment


class SequencerTestCase(unittest.TestCase):

    SEQUENCER = ['vdc', 'linear_feedback_shift', 'random']

    def test_output_01(self):
        n = 64
        k = 4

        some_simulation = LTFArray(
            weight_array=LTFArray.normal_weights(n, k),
            transform=LTFArray.transform_atf,
            combiner=LTFArray.combiner_xor
        )
        some_seed = [0] * n

        for sequencer_name in self.SEQUENCER:
            sequencer = getattr(NISTRandomnessExperiment, 'sequencer_%s' % sequencer_name)
            sequence = sequencer(some_simulation, some_seed, 100)
            for b in sequence:
                self.assertIn(b, [0, 1], "%s sequencer returned a sequence with element %s." %
                              (sequencer_name, b))
