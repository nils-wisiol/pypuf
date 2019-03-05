"""
Index of the first successful permutation for the correlation attack.
"""

from pypuf.experiments.experiment.correlation_attack import ExperimentCorrelationAttack
from pypuf.plots import PermutationIndexPlot
from pypuf.studies.base import Study


class BreakingLightweightSecureFig05(Study):
    SAMPLES_PER_GRAPH = 1000

    KS = [4, 4, 5, 6]
    TRAINING_SET_SIZES = [12000, 30000, 300000, 1000000]

    def __init__(self):
        super().__init__()
        self.result_plot = None

    def name(self):
        return 'breaking_lightweight_secure_fig_05'

    def plot(self):
        if not self.results:
            return

        if not self.result_plot:
            self.result_plot = PermutationIndexPlot(
                filename='figures/breaking_lightweight_secure_fig_05.pdf',
                results=self.results,
                group_by='N',
                group_labels={
                    str(self.TRAINING_SET_SIZES[i]):
                        "Perm. Index Dist. (k={}, {:,} CRPs)".format(self.KS[i], self.TRAINING_SET_SIZES[i])
                    for i in range(len(self.TRAINING_SET_SIZES))
                },
                #group_subplot_layout={
                #    self.TRAINING_SET_SIZES[0]: (1, 2, 1)
                #}
            )

        self.result_plot.plot()

    def experiments(self):
        experiments = []
        for idx, training_set_size in enumerate(self.TRAINING_SET_SIZES):
            for i in range(self.SAMPLES_PER_GRAPH):
                experiments.append(
                    ExperimentCorrelationAttack(
                        progress_log_prefix=self.name(),
                        n=64,
                        k=self.KS[idx],
                        N=training_set_size,
                        seed_instance=314159 + i,
                        seed_model=265358 + i,
                        seed_challenge=979323 + i,
                        seed_challenge_distance=846264 + i,
                    )
                )
        return experiments
