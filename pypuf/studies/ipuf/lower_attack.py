from matplotlib.pyplot import close
from seaborn import catplot

from pypuf.experiments.experiment.reliability_based_ipuf_lower_attack import ExperimentReliabilityBasedLowerIPUFLearning, Parameters
from pypuf.studies.base import Study

import os

class LowerIPUFAttackStudy(Study):

    def __init__(self):
        super().__init__(gpu_limit=2)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    def experiments(self):
        return [
            ExperimentReliabilityBasedLowerIPUFLearning(
                progress_log_name=None,
                parameters=Parameters(
                    n=n,
                    k=k,
                    seed=seed,
                    noisiness=noisiness,
                    num=N,
                    reps=R,
                    abort_delta=0.005
                )
            )
            for n in [64]
            for noisiness in [0.15]
            for k, N in [(8, 1000000)]
            for R in [11]
            for seed in range(10)
        ]

        """
            for n in [64]
            for noisiness in [.1, .25,]
            for k, N in [(1, 20*10**3),(2, 100*10**3),(4, 150*10**3),(6, 300*10**3),(8, 500*10**3)]
            for R in [11]
            for seed in [1, 42, 1337]
        """


    def plot(self):
        data = self.experimenter.results.copy()
        data['reps__noisiness'] = data.apply(
            lambda row: f'{row["reps"]}__{row["noisiness"]}', axis=1)
        data['accuracy1'] = data.apply(lambda row: max(row['accuracy'],
            1 - row['accuracy']), axis=1)
        for hue in ['', 'reps', 'noisiness']:
            grid = catplot(
                x='num',
                y='accuracy1',
                hue=hue or 'reps__noisiness',
                col='k',
                row='transform',
                data=data,
                kind='boxen',
            )
            grid.savefig(f'figures/{self.name()}{"." if hue else ""}{hue}.pdf')
            close(grid.fig)
