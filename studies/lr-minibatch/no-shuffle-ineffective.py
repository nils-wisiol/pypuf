"""
Success Rate for Logistic Regression With and Without Mini Batches

Experiments in this file demonstrate that mini batches in our LR are
ineffective when there is no shuffling.
"""
from pypuf.experiments.experimenter import Experimenter
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, CompoundTransformation
from pypuf.plots import SuccessRatePlot
from numpy.random import RandomState


samples_per_point = 50

definitions = [
    (64, 2, [300, 400, 500, 600, 750, 1000, 1500]),
    (64, 4, [1000, 1500, 2000, 5000, 10000]),
    (128, 2, [600, 750, 1000, 1500, 2000, 5000, 10000]),
    (128, 4, [1000, 2000, 5000, 7500, 10000, 12000, 15000]),
]

minibatch_sizes = [100, 250, 500, 1000, 2000]
group_labels = {}
group_labels['None'] = 'No Mini Batches'
for size in minibatch_sizes:
    group_labels[str(size)] = '%i per Mini Batch' % size


for (n, k, training_set_sizes) in definitions:
    experiments = []
    log = 'minibatch-no-shuffle-ineffective-%i-%i.log' % (n, k)
    e = Experimenter(log, experiments)
    for training_set_size in training_set_sizes:
        for i in range(samples_per_point):
            for minibatch_size in [None] + minibatch_sizes:
                if minibatch_size and minibatch_size > training_set_size: break
                experiments.append(
                    ExperimentLogisticRegression(
                        log_name=log,
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
                        convergance_decimals=1.5 if not minibatch_size else 2.9,
                    )
                )
    RandomState(seed=1).shuffle(experiments)

    result_plot = SuccessRatePlot(
        filename='figures/minibatch-no-shuffle-ineffective-%i-%i.pdf' % (n, k),
        results=e.results,
        group_by='minibatch_size',
        group_labels=group_labels,
    )


    def update_plot():
        result_plot.plot()


    e.update_callback = update_plot
    e.run()

    result_plot.plot()

