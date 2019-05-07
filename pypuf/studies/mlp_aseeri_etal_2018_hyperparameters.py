from matplotlib.pyplot import subplots
from matplotlib.ticker import FixedLocator, FuncFormatter
from seaborn import lineplot, scatterplot

from pypuf.studies.base import Study
from pypuf.experiments.experiment.mlp_tf import ExperimentMLPTensorflow, Parameters as Parameters_tf
from pypuf.experiments.experiment.mlp_skl import ExperimentMLPScikitLearn, Parameters as Parameters_skl
from pypuf.plots import get_estimator


class MLPAseeriEtAlHyperparameterStudy(Study):

    TRANSFORMATION = 'id'
    COMBINER = 'xor'
    PREPROCESSING = 'no'
    ACTIVATION = 'relu'
    PATIENCE = 10
    ITERATION_LIMIT = 100
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False

    SAMPLES_PER_POINT = {
        4: 7,
        5: 7,
        6: 7,
        7: 7,
        8: 7,
    }

    SIZES = [
        (64, 4, 0.4e6),
        (64, 5, 0.8e6),
        (64, 6, 2e6),
        (64, 7, 5e6),
        (64, 8, 20e6),
    ]

    PLOT_ESTIMATORS = ['mean', 'best', 'success=0.9']

    TOLERANCES = {
        4: [0.1],
        5: [0.1],
        6: [0.1],
        7: [0.1],
        8: [0.1],
    }

    LEARNING_RATES = {
        4: [0.005, 0.010, 0.015],
        5: [0.003, 0.0035, 0.004],
        6: [0.007, 0.0075, 0.008],
        7: [0.004, 0.0045, 0.005],
        8: [0.001, 0.0015, 0.002],
    }

    PENALTIES = {
        4: [0.00005, 0.0001, 0.00015],
        5: [0.00005, 0.0001, 0.00015],
        6: [0.00005, 0.0001, 0.00015],
        7: [0.00005, 0.0001, 0.00015],
        8: [0.00005, 0.0001, 0.00015],
    }

    BETAS_1 = {
        4: [0.85, 0.9, 0.95],
        5: [0.85, 0.9, 0.95],
        6: [0.85, 0.9, 0.95],
        7: [0.85, 0.9, 0.95],
        8: [0.85, 0.9, 0.95],
    }

    BETAS_2 = {
        4: [0.9985, 0.999, 0.9995],
        5: [0.9985, 0.999, 0.9995],
        6: [0.9985, 0.999, 0.9995],
        7: [0.9985, 0.999, 0.9995],
        8: [0.9985, 0.999, 0.9995],
    }

    REFERENCE_TIMES = {
        (64, 4): 19.2,
        (64, 5): 58,
        (64, 6): 7.4 * 60,
        (64, 7): 11.8 * 60,
        (64, 8): 23.3 * 60,
        (128, 4): 1.32 * 60,
        (128, 5): 5 * 60,
        (128, 6): 19 * 60,
        (128, 7): 1.5 * 60**2,
    }

    REFERENCE_MEAN_ACCURACY = {
        (64, 4): .9842,
        (64, 5): .9855,
        (64, 6): .9915,
        (64, 7): .9921,
        (64, 8): .9874,
        (128, 4): .9846,
        (128, 5): .9870,
        (128, 6): .9903,
        (128, 7): .99,
    }

    def experiments(self):
        experiments = []
        for (n, k, N) in self.SIZES:
            validation_frac = max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N
            layers = [2**k, 2**k, 2**k]
            for i in range(self.SAMPLES_PER_POINT[k]):
                for j, learning_rate in enumerate(self.LEARNING_RATES[k]):
                    number = i * (len(self.LEARNING_RATES[k])) + j
                    for penalty in self.PENALTIES[k]:
                        for beta_1 in self.BETAS_1[k]:
                            for beta_2 in self.BETAS_2[k]:
                                for tolerance in self.TOLERANCES[k]:
                                    experiments.append(
                                        ExperimentMLPScikitLearn(
                                            progress_log_prefix=None,
                                            parameters=Parameters_skl(
                                                seed_simulation=0x3 + number,
                                                seed_challenges=0x1415 + number,
                                                seed_model=0x9265 + number,
                                                seed_distance=0x3589 + number,
                                                n=n,
                                                k=k,
                                                N=int(N),
                                                validation_frac=validation_frac,
                                                transformation=self.TRANSFORMATION,
                                                combiner=self.COMBINER,
                                                preprocessing=self.PREPROCESSING,
                                                layers=layers,
                                                activation=self.ACTIVATION,
                                                learning_rate=learning_rate,
                                                penalty=penalty,
                                                beta_1=beta_1,
                                                beta_2=beta_2,
                                                tolerance=tolerance,
                                                patience=self.PATIENCE,
                                                iteration_limit=self.ITERATION_LIMIT,
                                                batch_size=1000 if k < 6 else 10000,
                                                print_learning=self.PRINT_LEARNING,
                                            )
                                        )
                                    )
                                    experiments.append(
                                        ExperimentMLPTensorflow(
                                            progress_log_prefix=None,
                                            parameters=Parameters_tf(
                                                seed_simulation=0x3 + number,
                                                seed_challenges=0x1415 + number,
                                                seed_model=0x9265 + number,
                                                seed_distance=0x3589 + number,
                                                n=n,
                                                k=k,
                                                N=int(N),
                                                validation_frac=validation_frac,
                                                transformation='id',
                                                combiner='xor',
                                                preprocessing=self.PREPROCESSING,
                                                layers=layers,
                                                activation='relu',
                                                learning_rate=learning_rate,
                                                penalty=penalty,
                                                beta_1=beta_1,
                                                beta_2=beta_2,
                                                tolerance=tolerance,
                                                patience=self.PATIENCE,
                                                iteration_limit=self.ITERATION_LIMIT,
                                                batch_size=1000 if k < 6 else 10000,
                                                termination_threshold=1.0,
                                                print_learning=self.PRINT_LEARNING,
                                            )
                                        )
                                    )
        return experiments

    def plot(self):
        self.plot_helper(
            name='Scikit-Learn',
            df=self.experimenter.results[self.experimenter.results['experiment'] == 'ExperimentMLPScikitLearn'],
        )
        self.plot_helper(
            name='Tensorflow',
            df=self.experimenter.results[self.experimenter.results['experiment'] == 'ExperimentMLPTensorflow'],
        )

    def plot_helper(self, name, df):
        ks = sorted(list(set(self.experimenter.results['k'])))

        format_time = lambda x, _ = None: '%.1fs' % x if x < 60 else '%.1fmin' % (x / 60)

        fig, axes = subplots(ncols=len(ks), nrows=len(self.PLOT_ESTIMATORS) + 2)
        fig.set_size_inches(8*len(ks), 6 * len(self.PLOT_ESTIMATORS))

        axes = axes.reshape((len(self.PLOT_ESTIMATORS) + 2, len(ks)))

        self.experimenter.results['distance'] = 1 - self.experimenter.results['accuracy']
        for j, k in enumerate(ks):
            data = df[df['k'] == k]

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
            axes[-1][j].set_xlabel('accuracy')
            axes[-1][j].set_ylabel('measured_time\nreference: %s' % format_time(self.REFERENCE_TIMES[(64, k)]))   # TODO fix for n = 128
            axes[-1][j].xaxis.set_major_locator(FixedLocator([
                .5,
                .3,
                .2,
                .1,
                .05,
                .02,
                .01,
                .005,
            ]))
            axes[-1][j].xaxis.set_major_formatter(FuncFormatter(lambda x, _: '%.3f' % (1 - x)))
            #axes[-1][j].yaxis.set_major_locator(FixedLocator(self.REFERENCE_TIMES.values()))
            axes[-1][j].yaxis.set_major_formatter(FuncFormatter(format_time))
            axes[-1][j].yaxis.set_minor_formatter(FuncFormatter(format_time))

        fig.subplots_adjust(hspace=.5, wspace=.5)
        fig.suptitle('Accuracy of {} Results on XOR Arbiter PUF'.format(name))

        fig.savefig('figures/{}_{}.pdf'.format(self.name(), name), bbox_inches='tight', pad_inches=.5)
