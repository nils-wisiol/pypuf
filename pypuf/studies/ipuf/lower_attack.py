import os
from matplotlib.pyplot import close
from seaborn import catplot
from pypuf.studies.base import Study
from pypuf.experiments.experiment.reliability_based_ipuf_lower_attack import \
    ExperimentReliabilityBasedLowerIPUFLearning, Parameters


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
                    k_up=k_up,
                    k_down=k_down,
                    seed=seed,
                    noisiness=noisiness,
                    N=N,
                    R=R,
                    eps=eps,
                    extra=extra,
                    abort_delta=0.0001,
                    max_tries=k_down,
                    gpu_id=1,
                )
            )
            for n in [64]
            for noisiness in [0.05]
            for k_up, k_down, extra, N in [
                (1, 2, 2, 100000),
                (1, 3, 2, 200000),
                (1, 4, 3, 300000),
                (1, 5, 3, 500000),
                (1, 6, 2, 300000),
                (1, 6, 4, 800000),
            ]
            for R in [11]
            for eps in [0.9]
            for seed in range(100)
        ]

    def plot(self):
        """
        data = self.experimenter.results.copy()
        data['reps__noisiness'] = data.apply(
            lambda row: f'{row["reps"]}__{row["noisiness"]}', axis=1)
        data['accuracy1'] = data.apply(lambda row: max(row['accuracy'], 1 - row['accuracy']), axis=1)
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
        """
        pass
