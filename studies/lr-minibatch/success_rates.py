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


SAMPLES_PER_POINT = 10

DEFINITIONS = [
    (64, 1, [300, 400, 500, 600, 750, 1000, 1500]),
    (64, 2, [5000, 10000, 10000, 12000, 15000, 20000]),
    (128, 2, [600, 750, 1000, 1500, 2000, 5000]),
    (128, 4, [15000, 20000, 50000, 100000, 200000]),
]

MINIBATCH_SIZES = [100, 250, 500, 1000, 2000]
GROUP_LABELS = {}
GROUP_LABELS['None'] = 'No Mini Batches'
for size in MINIBATCH_SIZES:
    GROUP_LABELS[str(size)] = '%i per Mini Batch' % size

plots = []


def update_plots():
    """
    Updates the result plot.
    """
    for p in plots:
        p.plot()


log_name = 'lr-minibatch-success-rates'
e = Experimenter(log_name)
for (n, k, training_set_sizes) in DEFINITIONS:
    for shuffle in [True, False]:
        plot = SuccessRatePlot(
            filename='figures/lr-minibatch' + ('' if shuffle else '-no') + '-shuffle-ineffective-%i-%i.pdf' % (n, k),
            results=e.results,
            group_by='minibatch_size',
            group_labels=GROUP_LABELS,
        )
        plots.append(plot)
        for training_set_size in training_set_sizes:
            for i in range(SAMPLES_PER_POINT):
                for minibatch_size in [None] + MINIBATCH_SIZES:
                    if minibatch_size and minibatch_size >= training_set_size:
                        break
                    experiment_id = e.queue(
                        ExperimentLogisticRegression(
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
                        )
                    )
                    plot.experiment_ids.append(experiment_id)


e.update_callback = update_plots
e.run(shuffle=True)

update_plots()
