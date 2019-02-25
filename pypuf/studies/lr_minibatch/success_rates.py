"""
Success Rate for Logistic Regression With and Without Mini Batches

This study examines the effect of mini batches in our LR learner.
Results are:

- Mini batches in our LR are ineffective when there is no shuffling.
"""
from pypuf.experiments.experimenter import Experimenter
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.plots import SuccessRatePlot
from pypuf.studies.base import Study


class LRMiniBatchSuccessRate(Study):
    SAMPLES_PER_POINT = 20

    DEFINITIONS = [
        # (64, 1, [5, 10, 30, 50, 70, 90, 100, 150, 200]),  # values okay
        # (64, 2, [200, 300, 400, 500, 600, 700, 800, 900, 1000]),  # values okay
        (64, 4, [5000, 10000, 20000, 25000, 35000, 50000, 100000, 200000]),
        (64, 5, [50000, 100000, 200000, 300000, 400000, 500000]),
    ]

    MINI_BATCH_SIZES = [5000, 10000]  # [20, 50, 100, 200] for n=64, k∈{1,2}
    GROUP_LABELS = {}
    GROUP_LABELS['None'] = 'No Mini Batches'

    def __init__(self):
        super().__init__()
        self.result_plots = []

        for size in self.MINI_BATCH_SIZES:
            self.GROUP_LABELS[str(size)] = '%i per Mini Batch' % size

    def name(self):
        return 'success_rate'

    def experiments(self):
        experiments = []
        for (n, k, training_set_sizes) in self.DEFINITIONS:
            for shuffle in [True, False]:
                plot = SuccessRatePlot(
                    filename='figures/lr-minibatch-' + ('shuffle' if shuffle else 'noshuffle') + '-success_rate-%i-%i.pdf' % (
                    n, k),
                    results=self.experimenter.results,
                    group_by='minibatch_size',
                    group_labels=self.GROUP_LABELS,
                )
                self.result_plots.append(plot)
                for training_set_size in training_set_sizes:
                    for i in range(self.SAMPLES_PER_POINT):
                        for minibatch_size in [None] + self.MINI_BATCH_SIZES:
                            if minibatch_size and minibatch_size >= training_set_size:
                                break

                            e = ExperimentLogisticRegression(
                                progress_log_prefix=None,
                                n=n,
                                k=k,
                                N=training_set_size,
                                seed_instance=314159 + i,
                                seed_model=265358 + i,
                                transformation=LTFArray.transform_id,
                                combiner=LTFArray.combiner_xor,
                                seed_challenge=979323 + i,
                                seed_chl_distance=846264 + i,
                                minibatch_size=minibatch_size,
                                convergance_decimals=1.5 if not minibatch_size else 2.7,
                                shuffle=shuffle,
                            )
                            experiments.append(e)
                            plot.experiment_ids.append(e.id)
        return experiments

    def plot(self):
        if not self.results:
            return

        for plot in self.result_plots:
            plot.plot()
