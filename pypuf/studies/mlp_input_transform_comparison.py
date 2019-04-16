"""
Hyperparameter Optimization: Testing Regularization methods and different structures of the MLP.
"""
from matplotlib.pyplot import subplots
from matplotlib.ticker import FuncFormatter, FixedLocator
from numpy.ma import arange
from seaborn import lineplot, scatterplot

from pypuf.experiments.experiment.mlp_tf import ExperimentMLPTensorflow, Parameters
from pypuf.plots import AccuracyPlotter, get_estimator
from pypuf.studies.base import Study
from numpy.random.mtrand import RandomState


class MLPInputTransformComparison(Study):

    CPU_LIMIT = 50
    SEED_RANGE = 2 ** 32
    SAMPLES_PER_POINT = 3
    TRANSFORMATIONS = ['atf', 'lightweight_secure', 'fixed_permutation']
    COMBINER = 'xor'
    BATCH_SIZE = 1000
    ITERATION_LIMIT = 10000
    PRINT_KERAS = False
    LOG_NAME = 'adam'

    DEFINITIONS = [
        [[6, 6], (0.001, 0.01), 0.9, 0.999, 2, 64, (2000, 2000), True, 14159, 26535, 89793, 23846],
        [[6, 6], (0.001, 0.01), 0.9, 0.99, 2, 64, (2000, 2000), True, 26433, 83279, 50288, 41971],
        [[6, 6], (0.001, 0.01), 0.9, 0.9999, 2, 64, (2000, 2000), True, 69399, 37510, 58209, 74944],
        [[6, 6], (0.001, 0.01), 0.85, 0.999, 2, 64, (2000, 2000), True, 59230, 78164, 6286, 20899],
        [[6, 6], (0.001, 0.01), 0.85, 0.99, 2, 64, (2000, 2000), True, 86280, 34825, 34211, 70679],
        [[6, 6], (0.001, 0.01), 0.85, 0.9999, 2, 64, (2000, 2000), True, 82148, 8651, 32823, 6647],
        [[6, 6], (0.001, 0.01), 0.95, 0.999, 2, 64, (2000, 2000), True, 9384, 46095, 50582, 23172],
        [[6, 6], (0.001, 0.01), 0.95, 0.99, 2, 64, (2000, 2000), True, 53594, 8128, 48111, 74502],
        [[6, 6], (0.001, 0.01), 0.95, 0.9999, 2, 64, (2000, 2000), True, 84102, 70193, 85211, 5559],

        [[32, 32], (0.001, 0.01), 0.9, 0.999, 4, 64, (400000, 400000), True, 64462, 29489, 54930, 38196],
        [[32, 32], (0.001, 0.01), 0.9, 0.99, 4, 64, (400000, 400000), True, 44288, 10975, 66593, 34461],
        [[32, 32], (0.001, 0.01), 0.9, 0.9999, 4, 64, (400000, 400000), True, 28475, 64823, 37867, 83165],
        [[32, 32], (0.001, 0.01), 0.85, 0.999, 4, 64, (400000, 400000), True, 27120, 19091, 45648, 56692],
        [[32, 32], (0.001, 0.01), 0.85, 0.99, 4, 64, (400000, 400000), True, 34603, 48610, 45432, 66482],
        [[32, 32], (0.001, 0.01), 0.85, 0.9999, 4, 64, (400000, 400000), True, 13393, 60726, 2491, 41273],
        [[32, 32], (0.001, 0.01), 0.95, 0.999, 4, 64, (400000, 400000), True, 72458, 70066, 6315, 58817],
        [[32, 32], (0.001, 0.01), 0.95, 0.99, 4, 64, (400000, 400000), True, 48815, 20920, 96282, 92540],
        [[32, 32], (0.001, 0.01), 0.95, 0.9999, 4, 64, (400000, 400000), True, 91715, 36436, 78925, 90360],

        [[64, 64, 64], (0.001, 0.01), 0.9, 0.999, 6, 64, (2000000, 2000000), True, 1133, 5305, 48820, 46652],
        [[64, 64, 64], (0.001, 0.01), 0.9, 0.99, 6, 64, (2000000, 2000000), True, 13841, 46951, 94151, 16094],
        [[64, 64, 64], (0.001, 0.01), 0.9, 0.9999, 6, 64, (2000000, 2000000), True, 33057, 27036, 57595, 91953],
        [[64, 64, 64], (0.001, 0.01), 0.85, 0.999, 6, 64, (2000000, 2000000), True, 9218, 61173, 81932, 61179],
        [[64, 64, 64], (0.001, 0.01), 0.85, 0.99, 6, 64, (2000000, 2000000), True, 31051, 18548, 7446, 23799],
        [[64, 64, 64], (0.001, 0.01), 0.85, 0.9999, 6, 64, (2000000, 2000000), True, 62749, 56735, 18857, 52724],
        [[64, 64, 64], (0.001, 0.01), 0.95, 0.999, 6, 64, (2000000, 2000000), True, 89122, 79381, 83011, 94912],
        [[64, 64, 64], (0.001, 0.01), 0.95, 0.99, 6, 64, (2000000, 2000000), True, 31415, 92653, 58979, 32384],
        [[64, 64, 64], (0.001, 0.01), 0.95, 0.9999, 6, 64, (2000000, 2000000), True, 62643, 38327, 95028, 84197],
        #[[6, 6], 2, 64, (1000, 8000), True, 0x993742f6, 0x5cfed54, 0xb275a0c, 0x8917e5bb],
        #[[24, 24], 4, 64, (50000, 50000), True, None, None, None, None],  # 0xa4286b0d, 0xe52ff1c7, 0xf7012eba, 0x1c227d87],
        #[[96, 96], 6, 64, (2500000, 10000000), True, 0x2aa9c0be, 0x811ef1a, 0x13a8b53d, 0x8b45e5e],
        #[2, 64, (1000, 8000), False, 0x49b18fe9, 0x2cfc7ba7, 0xc0071fa6, 0xd7f9f178],
        #[4, 64, (25000, 200000), False, 0x49b18fe9, 0x2cfc7ba7, 0xc0071fa6, 0xd7f9f178],
        #[6, 64, (625000, 5000000), False, 0x32ca7e39, 0xac3e3392, 0xd4bb6c20, 0xa3eed0a8],
    ]

    PLOT_ESTIMATORS = ['mean', 'best', 'success=0.9']

    def __init__(self):
        super().__init__(cpu_limit=self.CPU_LIMIT)
        self.result_plots = {}
        for layers, learning_rates, beta_1, beta_2, k, n, (min_CRPs, max_CRPs), \
            preprocess, seed_s, seed_c, seed_m, seed_a in self.DEFINITIONS:
            self.result_plots[(k, n, 'mean')] = AccuracyPlotter(
                min_tick=min_CRPs,
                max_tick=max_CRPs,
                group_by='learning_rate',
                group_by_ex='transformation',
                estimator='mean',
                grid=True,
            )
            q = 1.0
            self.result_plots[(k, n, 'quantile={}'.format(q))] = AccuracyPlotter(
                min_tick=min_CRPs,
                max_tick=max_CRPs,
                group_by='learning_rate',
                group_by_ex='transformation',
                estimator=('quantile', q),
                grid=True,
            )
            """
            p = 0.7
            self.result_plots[(k, n, 'success={}'.format(p))] = AccuracyPlotter(
                min_tick=min_CRPs,
                max_tick=max_CRPs,
                group_by='learning_rate',
                group_by_ex='transformation',
                estimator=('success', p),
                grid=False,
            )
            """

    def name(self):
        return self.LOG_NAME

    def experiments(self):
        experiments = []

        for layers, (min_lr, max_lr), beta_1, beta_2, k, n, (min_CRPs, max_CRPs), \
            preprocess, seed_s, seed_c, seed_m, seed_a in self.DEFINITIONS:
            seed_s = seed_s or RandomState().randint(self.SEED_RANGE)
            seed_c = seed_c or RandomState().randint(self.SEED_RANGE)
            seed_m = seed_m or RandomState().randint(self.SEED_RANGE)
            seed_a = seed_a or RandomState().randint(self.SEED_RANGE)
            nums = len(range(min_CRPs, max_CRPs + min_CRPs, min_CRPs))
            for s in range(self.SAMPLES_PER_POINT):
                for i, N in enumerate(range(min_CRPs, max_CRPs + min_CRPs, min_CRPs)):
                    for lr in arange(min_lr, max_lr + min_lr, min_lr):
                        for transformation in self.TRANSFORMATIONS:
                            parameters_experiment = Parameters(
                                seed_simulation=(seed_s + s * nums + i) % self.SEED_RANGE,
                                seed_challenges=(seed_c + s * nums + i) % self.SEED_RANGE,
                                seed_model=(seed_m + s * nums + i) % self.SEED_RANGE,
                                seed_distance=(seed_a + s * nums + i) % self.SEED_RANGE,
                                layers=layers,
                                learning_rate=lr,
                                beta_1=beta_1,
                                beta_2=beta_2,
                                N=N,
                                n=n,
                                k=k,
                                transformation=transformation,
                                combiner=self.COMBINER,
                                batch_size=self.BATCH_SIZE,
                                iteration_limit=self.ITERATION_LIMIT,
                                print_keras=self.PRINT_KERAS,
                            )
                            experiment = ExperimentMLPTensorflow(
                                progress_log_prefix=None,
                                parameters=parameters_experiment,
                            )
                            experiments.append(experiment)
        return experiments

    """
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
    """

    def plot(self):
        ks = sorted(list(set(self.experimenter.results['k'])))

        fig, axes = subplots(ncols=len(ks), nrows=len(self.PLOT_ESTIMATORS) + 2)
        fig.set_size_inches(20 * len(ks), 30 * len(self.PLOT_ESTIMATORS))

        axes = axes.reshape((len(self.PLOT_ESTIMATORS) + 2, len(ks)))

        self.experimenter.results['distance'] = 1 - self.experimenter.results['accuracy']
        for j, k in enumerate(ks):
            data = self.experimenter.results[self.experimenter.results['k'] == k]

            for i, estimator in enumerate(self.PLOT_ESTIMATORS):
                lineplot(
                    x='learning_rate',
                    y='accuracy',
                    hue='beta_1',
                    style='beta_2',
                    data=data,
                    ax=axes[i][j],
                    legend='full',
                    estimator=get_estimator(estimator),
                    ci=None,
                )
                scatterplot(
                    x='learning_rate',
                    y='accuracy',
                    hue='beta_1',
                    style='beta_2',
                    data=data,
                    ax=axes[i][j],
                    legend='full',
                )
                axes[i][j].set_ylabel('k=%i\naccuracy %s' % (k, estimator))
                axes[i][j].set_ylim((.45, 1.05))

            scatterplot(
                x='learning_rate',
                y='accuracy',
                hue='beta_1',
                style='beta_2',
                data=data,
                ax=axes[-2][j],
                legend='full',
            )

            lineplot(
                x='distance',
                y='measured_time',
                # style='tol',
                hue='beta_1',
                data=data,
                ax=axes[-1][j],
                legend='full',
                estimator='mean',
            )

            axes[-1][j].set_xscale('log')
            axes[-1][j].set_yscale('log')
            # axes[-1].xaxis.set_major_locator(MultipleLocator(.2))
            # axes[-1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axes[-1][j].yaxis.set_major_locator(FixedLocator([
                # References for:
                19.2,  # k = 4
                58,  # k = 5
                7.4 * 60,  # k = 6
                11.8 * 60,  # k = 7
                23.3 * 60,  # k = 8
            ]))
            axes[-1][j].set_xlabel('accuracy')
            axes[-1][j].xaxis.set_major_formatter(FuncFormatter(lambda x, _: '%.3f' % (1 - x)))
            axes[-1][j].xaxis.set_minor_formatter(FuncFormatter(lambda x, _: '%.3f' % (1 - x)))
            axes[-1][j].yaxis.set_major_formatter(FuncFormatter(
                lambda x, _: '%3.1fs' % x if x < 60 else '%2.1fmin' % (x / 60)))

        fig.subplots_adjust(hspace=.5, wspace=.5)
        fig.suptitle('Accuracy of Tensorflow Results on XOR Arbiter PUF')

        fig.savefig('figures/{}.pdf'.format(self.name()), bbox_inches='tight', pad_inches=.5)

