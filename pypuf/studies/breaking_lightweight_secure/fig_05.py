"""
Index of the first successful permutation for the correlation attack.
"""

from pypuf.experiments.experiment.correlation_attack import ExperimentCorrelationAttack, Parameters
from pypuf.plots import PermutationIndexPlot
from pypuf.studies.base import Study


class BreakingLightweightSecureFig05(Study):
    SAMPLES_PER_GRAPH = 100

    KS = [4, 4, 5, 6]
    TRAINING_SET_SIZES = [12000, 30000, 300000, 1000000]

    def __init__(self):
        super().__init__()
        self.result_plot = None

    def name(self):
        return 'breaking_lightweight_secure_fig_05'

    def plot(self):
        if self.experimenter.results.empty:
            return

        self.result_plot.plot(self.experimenter.results)

    def experiments(self):
        experiments = []

        self.result_plot = PermutationIndexPlot(
            filename='figures/breaking_lightweight_secure_fig_05.pdf',
            group_by='N',
            group_labels={
                self.TRAINING_SET_SIZES[i]:
                    "Perm. Index Dist. (k={}, {:,} CRPs)".format(self.KS[i], self.TRAINING_SET_SIZES[i])
                for i in range(len(self.TRAINING_SET_SIZES))
            },
            group_subplot_layout={
                self.TRAINING_SET_SIZES[0]: (4, 2, 1),
                self.TRAINING_SET_SIZES[1]: (4, 2, 2),
                self.TRAINING_SET_SIZES[2]: (4, 1, 2),
                self.TRAINING_SET_SIZES[3]: (4, 1, 3),
            }, w=3.34, h=3.9
        )

        for idx, training_set_size in enumerate(self.TRAINING_SET_SIZES):
            for i in range(self.SAMPLES_PER_GRAPH):
                experiments.append(
                    ExperimentCorrelationAttack(
                        progress_log_prefix=self.name(),
                        parameters=Parameters(
                            n=64,
                            k=self.KS[idx],
                            N=training_set_size,
                            seed_instance=314159 + i,
                            seed_model=265358 + i,
                            seed_challenge=979323 + i,
                            seed_distance=846264 + i,
                            lr_iteration_limit=1000,
                            mini_batch_size=10000,
                            convergence_decimals=2,
                            shuffle=True
                        )
                    )
                )
        return experiments
