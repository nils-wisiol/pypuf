"""
Hyperparameter Optimization: Testing Regularization methods and different structures of the MLP.
"""
from pypuf.experiments.experiment.mlp_keras import ExperimentMLP, Parameters
from pypuf.plots import AccuracyPlotter
from pypuf.studies.base import Study
from numpy.random.mtrand import RandomState


class MLPInputTransformComparison(Study):

    CPU_LIMIT = 4
    SEED_RANGE = 2 ** 32
    SAMPLES_PER_POINT = 3
    TRANSFORMATIONS = ['atf', 'lightweight_secure', 'fixed_permutation']
    COMBINER = 'xor'
    BATCH_SIZE = 1000
    ITERATION_LIMIT = 10000
    PRINT_KERAS = False
    LOG_NAME = 'test_mlp'

    DEFINITIONS = [
        [[5, 5], 2, 64, (1000, 8000), True, None, None, None, None],
        [[6, 6], 2, 64, (1000, 8000), True, None, None, None, None],
        [[7, 7], 2, 64, (1000, 8000), True, None, None, None, None],
        [[8, 8], 2, 64, (1000, 8000), True, None, None, None, None],
        [[9, 9], 2, 64, (1000, 8000), True, None, None, None, None],
        [[10, 10], 2, 64, (1000, 8000), True, None, None, None, None],
        [[10, 4], 2, 64, (1000, 8000), True, None, None, None, None],
        [[8, 4], 2, 64, (1000, 8000), True, None, None, None, None],
        [[6, 4], 2, 64, (1000, 8000), True, None, None, None, None],
        [[10, 6], 2, 64, (1000, 8000), True, None, None, None, None],
        [[8, 6], 2, 64, (1000, 8000), True, None, None, None, None],
        [[10, 8], 2, 64, (1000, 8000), True, None, None, None, None],
        #[[6, 6], 2, 64, (1000, 8000), True, 0x993742f6, 0x5cfed54, 0xb275a0c, 0x8917e5bb],
        #[[24, 24], 4, 64, (50000, 200000), True, None, None, None, None],  # 0xa4286b0d, 0xe52ff1c7, 0xf7012eba, 0x1c227d87],
        #[[96, 96], 6, 64, (2500000, 10000000), True, 0x2aa9c0be, 0x811ef1a, 0x13a8b53d, 0x8b45e5e],
        #[2, 64, (1000, 8000), False, 0x49b18fe9, 0x2cfc7ba7, 0xc0071fa6, 0xd7f9f178],
        #[4, 64, (25000, 200000), False, 0x49b18fe9, 0x2cfc7ba7, 0xc0071fa6, 0xd7f9f178],
        #[6, 64, (625000, 5000000), False, 0x32ca7e39, 0xac3e3392, 0xd4bb6c20, 0xa3eed0a8],
    ]

    def __init__(self):
        super().__init__(cpu_limit=self.CPU_LIMIT)
        self.result_plots = {}

    def name(self):
        return self.LOG_NAME

    def experiments(self):
        experiments = []

        for layers, k, n, (min_CRPs, max_CRPs), preprocess, seed_s, seed_c, seed_m, seed_a in self.DEFINITIONS:
            self.result_plots[(k, n, 'mean')] = AccuracyPlotter(
                min_tick=min_CRPs,
                max_tick=max_CRPs,
                group_by='transformation',
                estimator='mean',
                grid=True,
            )
            q = 0.9
            self.result_plots[(k, n, 'quantile={}'.format(q))] = AccuracyPlotter(
                min_tick=min_CRPs,
                max_tick=max_CRPs,
                group_by='transformation',
                estimator=('quantile', q),
                grid=True,
            )
            p = 0.7
            self.result_plots[(k, n, 'success={}'.format(p))] = AccuracyPlotter(
                min_tick=min_CRPs,
                max_tick=max_CRPs,
                group_by='transformation',
                estimator=('success', p),
                grid=False,
            )

            seed_s = seed_s or RandomState().randint(self.SEED_RANGE)
            seed_c = seed_c or RandomState().randint(self.SEED_RANGE)
            seed_m = seed_m or RandomState().randint(self.SEED_RANGE)
            seed_a = seed_a or RandomState().randint(self.SEED_RANGE)
            nums = len(range(min_CRPs, max_CRPs + min_CRPs, min_CRPs))
            for s in range(self.SAMPLES_PER_POINT):
                for i, N in enumerate(range(min_CRPs, max_CRPs + min_CRPs, min_CRPs)):
                    for transformation in self.TRANSFORMATIONS:
                        parameters_experiment = Parameters(
                            layers=layers,
                            N=N,
                            n=n,
                            k=k,
                            transformation=transformation,
                            combiner=self.COMBINER,
                            preprocess=preprocess,
                            batch_size=self.BATCH_SIZE,
                            iteration_limit=self.ITERATION_LIMIT,
                            print_keras=self.PRINT_KERAS,
                            seed_simulation=(seed_s + s*nums + i) % self.SEED_RANGE,
                            seed_challenges=(seed_c + s*nums + i) % self.SEED_RANGE,
                            seed_model=(seed_m + s*nums + i) % self.SEED_RANGE,
                            seed_accuracy=(seed_a + s*nums + i) % self.SEED_RANGE,
                        )
                        experiment = ExperimentMLP(
                            progress_log_prefix=None,
                            parameters=parameters_experiment,
                        )
                        experiments.append(experiment)
        return experiments

    def plot(self):
        for (k, n, plotter) in self.result_plots.keys():
            results = self.experimenter.results
            results = results.loc[(results['n'] == n) & (results['k'] == k)]
            if results.empty:
                continue
            self.result_plots[(k, n, plotter)].get_data_frame(results)
            self.result_plots[(k, n, plotter)].create_plot(
                save_path='figures/{}_k={}_n={}_{}.pdf'.format(self.name(), k, n, plotter))
        return
