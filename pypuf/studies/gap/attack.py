from matplotlib.pyplot import close
from seaborn import catplot

from pypuf.experiments.experiment.reliability_based_cmaes import ExperimentReliabilityBasedCMAES, Parameters
from pypuf.studies.base import Study


class ReliabilityAttackStudy(Study):

    def experiments(self):
        return [
            ExperimentReliabilityBasedCMAES(
                progress_log_name=None,
                parameters=Parameters(
                    n=n,
                    k=k,
                    seed_instance=seed,
                    seed_model=seed + 1,
                    seed_challenges=seed,
                    transform=transform,
                    combiner='xor',
                    noisiness=noisiness,
                    num=N,
                    reps=R,
                    abort_delta=0.005
                )
            )
            for n in [64]
            for transform in ['atf', 'lightweight_secure', 'fixed_permutation']
            for noisiness in [.1, .25]
            for k, N in [(1, 20*10**3), (2, 100*10**3), (4, 150*10**3), (6, 200*10**3), (8, 300*10**3)]
            for R in [3, 5, 11, 19, 49]
            for seed in [1, 42, 1337]
        ]

    """

            for n in [64]
            for k in [4]
            for transform in ['atf']
            for noisiness in [.25]
            for N in [150*10**3]
            for R in [11]
            for seed in [1]
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
