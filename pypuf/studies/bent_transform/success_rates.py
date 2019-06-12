from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression, Parameters
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, CompoundTransformation
from pypuf.plots import SuccessRatePlot
from pypuf.studies.base import Study


class SuccessRatesStudy(Study):
    """
    Success Rate for Logistic Regression on 64-Bit, 4-XOR Arbiter PUF
    """

    SAMPLES_PER_POINT = 100
    SUCCESS_THRESHOLD = .7
    SHUFFLE = True

    COMPRESSION = False

    INPUT_TRANSFORMATIONS = {
        'transform_atf': LTFArray.transform_atf,
        'transform_fixed_permutation': LTFArray.transform_fixed_permutation,
        'transform_random': LTFArray.transform_random,
        'transform_ipmod2_31415': CompoundTransformation(
            generator=LTFArray.generate_bent_transform,
            args=(64, 4, 31415),
            name='transform_ipmod2_31415',
        )
    }

    PRETTY_TRANSFORMATION_NAMES = {
        'transform_atf': 'Classic',
        'transform_fixed_permutation': 'Permutation-Based',
        'transform_random': 'Pseudorandom',
        'transform_ipmod2_31415': 'IP Mod 2 (Seed 31415)',
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
        'transform_ipmod2_31415': [
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

    def plot(self):
        results = self.experimenter.results

        fig = SuccessRatePlot(
            filename='figures/%s.pdf' % self.name(),
            group_by='transformation_name',
            group_labels=self.PRETTY_TRANSFORMATION_NAMES
        )
        fig.plot(results)

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
