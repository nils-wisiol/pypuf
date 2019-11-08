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
                    pop_size=pop_size,
                    abort_delta=0.005,
                    abort_iter=10,
                )
            )
            for n in [64]
            for k in [4]
            for transform in ['atf']
            for noisiness in [.25]
            for N in [150*10**3]
            for R in [11]
            for pop_size in [20]
            for seed in [1]
        ]

    """
        for n in [64]
        for k in [1, 2, 3, 4, 5, 6]
        for transform in ['atf', 'id', 'lightweight_secure', 'fixed_permutation']
        for noisiness in [.01, .1, .25, .5]
        for N in [10**4, 10**5, 5 * 10**5, 10**6]
        for R in [3, 5, 11, 19, 49]
        for pop_size in [20, 50, 90]
        for seed in range(10)
    """

    def plot(self):
        data = self.experimenter.results.copy()
        data['pop_size__reps__noisiness'] = data.apply(
            lambda row: f'{row["pop_size"]}__{row["reps"]}__{row["noisiness"]}', axis=1)
        data['accuracy1'] = data.apply(lambda row: max(row['accuracy'],
            1 - row['accuracy']), axis=1)
        for hue in ['', 'pop_size', 'reps', 'noisiness']:
            grid = catplot(
                x='num',
                y='accuracy1',
                hue=hue or 'pop_size__reps__noisiness',
                col='k',
                row='transform',
                data=data,
                kind='boxen',
            )
            grid.savefig(f'figures/{self.name()}{"." if hue else ""}{hue}.pdf')
            close(grid.fig)
