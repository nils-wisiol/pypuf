"""
Success Rate for Logistic Regression on 64-Bit, 4-XOR Arbiter PUF

Success rate of logistic regression attacks on simulated XOR Arbiter
PUFs with 64-bit arbiter chains and four arbiter chains each, based on at least
250 samples per data point shown. Accuracies better than 70% are considered
success (but cf. Figure 4). Four different designs are shown: of the four arbiter
chains in each instance, an input transform is used that transforms zero, one,
two, and three challenges pseudorandomly, keeping the remaining challenges
unmodified. The success rate decreases when the number of arbiter chains with
pseudorandom challenges is increased. The case with 4 pseudorandom sub-
challenges is not shown as it coincides with the results for 3 pseudorandom
challenges. Note the log-scale on the x-axis.
"""
import inspect
import sys

from pypuf.experiments.experimenter import Experimenter
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, CompoundTransformation
from pypuf.plots import SuccessRatePlot
from pypuf.studies.base import Study


class BreakingLightweightSecureFig03(Study):

    SAMPLES_PER_POINT = 250
    SUCCESS_THRESHOLD = .7

    INPUT_TRANSFORMATIONS = [
        LTFArray.transform_atf,
        CompoundTransformation(
            LTFArray.generate_stacked_transform, (LTFArray.transform_random, 1, LTFArray.transform_atf)),
        CompoundTransformation(
            LTFArray.generate_stacked_transform, (LTFArray.transform_random, 2, LTFArray.transform_atf)),
        CompoundTransformation(
            LTFArray.generate_stacked_transform, (LTFArray.transform_random, 3, LTFArray.transform_atf)),
    ]

    TRAINING_SET_SIZES = [
        [
            1000,
            2000,
            5000,
            10000,
            12000,
            15000,
            20000,
            30000,
            50000,
            100000,
            200000,
            1000000,
        ],
        [
            1000,
            2000,
            5000,
            10000,
            20000,
            30000,
            40000,
            50000,
            100000,
            200000,
            300000,
            1000000,
        ],
        [
            2000,
            5000,
            10000,
            20000,
            40000,
            50000,
            60000,
            100000,
            200000,
            400000,
            600000,
            1000000,
        ],
        [
            2000,
            5000,
            20000,
            40000,
            50000,
            60000,
            80000,
            100000,
            200000,
            400000,
            600000,
            800000,
            1000000,
            1500000,
            2000000,
        ],
    ]

    def __init__(self):
        super().__init__()
        self.result_plot = None

    def name(self):
        return 'breaking_lightweight_secure_fig_03'

    def plot(self):
        if not self.results:
            return

        if not self.result_plot:
            self.result_plot = SuccessRatePlot(
                filename='figures/breaking_lightweight_secure_fig_03.pdf',
                results=self.results,
                group_by='transformation',
                group_labels={
                    'transform_atf': 'Classic',
                    'transform_stack_random_nn1_atf': '1 Pseudorandom Sub-Challenge',
                    'transform_stack_random_nn2_atf': '2 Pseudorandom Sub-Challenges',
                    'transform_stack_random_nn3_atf': '3 Pseudorandom Sub-Challenges',
                }
            )

        self.result_plot.plot()

    def experiments(self):
        experiments = []
        for idx, transformation in enumerate(self.INPUT_TRANSFORMATIONS):
            for training_set_size in self.TRAINING_SET_SIZES[idx]:
                for i in range(self.SAMPLES_PER_POINT):
                    experiments.append(
                        ExperimentLogisticRegression(
                            progress_log_prefix=self.name(),
                            n=64,
                            k=4,
                            N=training_set_size,
                            seed_instance=314159 + i,
                            seed_model=265358 + i,
                            transformation=transformation,
                            combiner=LTFArray.combiner_xor,
                            seed_challenge=979323 + i,
                            seed_chl_distance=846264 + i,
                        )
                    )
        return experiments
