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
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression, Parameters
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, CompoundTransformation
from pypuf.plots import SuccessRatePlot
from pypuf.studies.base import Study


class BreakingLightweightSecureFig03(Study):

    SAMPLES_PER_POINT = 100
    SUCCESS_THRESHOLD = .7
    SHUFFLE = True

    INPUT_TRANSFORMATIONS = {
        'transform_atf': LTFArray.transform_atf,
        'transform_stack_random_nn1_atf': CompoundTransformation(
            LTFArray.generate_stacked_transform, (LTFArray.transform_random, 1, LTFArray.transform_atf)),
        'transform_stack_random_nn2_atf': CompoundTransformation(
            LTFArray.generate_stacked_transform, (LTFArray.transform_random, 2, LTFArray.transform_atf)),
        'transform_stack_random_nn3_atf': CompoundTransformation(
            LTFArray.generate_stacked_transform, (LTFArray.transform_random, 3, LTFArray.transform_atf)),
        'transform_fixed_permutation': LTFArray.transform_fixed_permutation,
        'transform_lightweight_secure': LTFArray.transform_lightweight_secure,
        'transform_random': LTFArray.transform_random,
    }

    PRETTY_TRANSFORMATION_NAMES = {
        'transform_atf': 'Classic',
        'transform_stack_random_nn1_atf': '1 Pseudorandom Sub-Challenge',
        'transform_stack_random_nn2_atf': '2 Pseudorandom Sub-Challenges',
        'transform_stack_random_nn3_atf': '3 Pseudorandom Sub-Challenges',
        'transform_fixed_permutation': 'Permutation-Based',
        'transform_lightweight_secure': 'Lightweight Secure',
        'transform_random': 'Pseudorandom'
    }

    TRAINING_SET_SIZES = {
        'transform_atf': [
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
        'transform_stack_random_nn1_atf': [
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
        'transform_stack_random_nn2_atf': [
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
        'transform_stack_random_nn3_atf': [
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
        'transform_fixed_permutation': [
            2000,
            12000,
            30000,
            100000,
            150000,
            400000,
            600000,
            1000000,
        ],
        'transform_lightweight_secure': [
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
        'transform_random': [
            2000,
            12000,
            30000,
            100000,
            150000,
            400000,
            600000,
            1000000,
        ],
    }

    def __init__(self):
        super().__init__()
        self.result_plot = None

    def name(self):
        return 'breaking_lightweight_secure_fig_03'

    def plot(self):
        results = self.experimenter.results

        fig_03 = SuccessRatePlot(
            filename='figures/breaking_lightweight_secure_fig_03.pdf',
            group_by='transformation_name',
            group_labels=self.PRETTY_TRANSFORMATION_NAMES
        )
        fig_03.plot(results[results['transformation_name'].isin([
            'transform_atf',
            'transform_stack_random_nn1_atf',
            'transform_stack_random_nn2_atf',
            'transform_stack_random_nn3_atf',
        ])])

        fig_06 = SuccessRatePlot(
            filename='figures/breaking_lightweight_secure_fig_06.pdf',
            group_by='transformation_name',
            group_labels=self.PRETTY_TRANSFORMATION_NAMES
        )
        fig_06.plot(results[results['transformation_name'].isin([
            'transform_atf',
            'transform_fixed_permutation',
            'transform_lightweight_secure',
            'transform_random',
        ])])

    def experiments(self):
        experiments = []
        for idx, transformation in self.INPUT_TRANSFORMATIONS.items():
            for training_set_size in self.TRAINING_SET_SIZES[idx]:
                for i in range(self.SAMPLES_PER_POINT):
                    experiments.append(
                        ExperimentLogisticRegression(
                            progress_log_prefix=None,
                            parameters=Parameters(
                                n=64,
                                k=4,
                                N=training_set_size,
                                seed_instance=314159 + i,
                                seed_model=265358 + i,
                                transformation=transformation,
                                combiner='xor',
                                seed_challenge=979323 + i,
                                seed_distance=846264 + i,
                                convergence_decimals=2,
                                mini_batch_size=0,
                                shuffle=False,
                            )
                        )
                    )
        return experiments
