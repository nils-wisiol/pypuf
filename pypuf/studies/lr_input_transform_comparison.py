from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression, Parameters
from pypuf.studies.base import Study
from pypuf.plots import AccuracyPlotter

from numpy.random.mtrand import RandomState


class LRInputTransformComparison(Study):

    CPU_LIMIT = 1
    SEED_RANGE = 2 ** 32
    SAMPLES_PER_POINT = 20
    TRANSFORMATIONS = ['id', 'atf', 'aes_substitution', 'lightweight_secure', 'random']
    COMBINER = 'xor'
    EXPERIMENTER_CALLBACK_MIN_PAUSE = 30
    LOG_NAME = 'lr_input_transform_comparison'

    DEFINITIONS = [
        [1, 64, (100, 2000), 0x993742f6, 0x5cfed54, 0xb275a0c, 0x8917e5bb],
        [2, 64, (200, 4000), 0x49fbc871, 0xd08a5b70, 0x515fe20f, 0xf3e0dacb],
        [3, 64, (1000, 20000), 0x49b18fe9, 0x2cfc7ba7, 0xc0071fa6, 0xd7f9f178],
        [4, 64, (5000, 100000), 0xa4286b0d, 0xe52ff1c7, 0xf7012eba, 0x1c227d87],
        [5, 64, (25000, 500000), 0x32ca7e39, 0xac3e3392, 0xd4bb6c20, 0xa3eed0a8],
        [6, 64, (125000, 2500000), 0x2aa9c0be, 0x811ef1a, 0x13a8b53d, 0x8b45e5e],
    ]

    def __init__(self):
        super().__init__(cpu_limit=self.CPU_LIMIT)
        self.result_plots = {}

    def name(self):
        return self.LOG_NAME

    def experiments(self):
        experiments = []

        for k, n, (min_CRPs, max_CRPs), seed_s, seed_c, seed_m, seed_a in self.DEFINITIONS:
            self.result_plots[(k, n, 'mean')] = AccuracyPlotter(
                min_tick=min_CRPs,
                max_tick=max_CRPs,
                group_by='transformation',
                estimator='mean',
                grid=True,
            )
            self.result_plots[(k, n, 'quantile')] = AccuracyPlotter(
                min_tick=min_CRPs,
                max_tick=max_CRPs,
                group_by='transformation',
                estimator=('quantile', 0.67),
                grid=True,
            )
            self.result_plots[(k, n, 'success')] = AccuracyPlotter(
                min_tick=min_CRPs,
                max_tick=max_CRPs,
                group_by='transformation',
                estimator=('success', 0.9),
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
                            N=N,
                            k=k,
                            n=n,
                            transformation=transformation,
                            combiner=self.COMBINER,
                            seed_instance=(seed_s + s * nums + i) % self.SEED_RANGE,
                            seed_challenge=(seed_c + s * nums + i) % self.SEED_RANGE,
                            seed_model=(seed_m + s * nums + i) % self.SEED_RANGE,
                            seed_distance=(seed_a + s * nums + i) % self.SEED_RANGE,
                            mini_batch_size=N,
                            convergence_decimals=4,
                            shuffle=False,
                        )
                        experiment = ExperimentLogisticRegression(
                            progress_log_prefix=None,
                            parameters=parameters_experiment
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
                save_path='figures/{}_k={}_n={}_{}.pdf'.format(plotter, k, n, self.name()))
        return
