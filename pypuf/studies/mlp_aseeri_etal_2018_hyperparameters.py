from re import findall
from math import log10
from matplotlib.pyplot import subplots
from matplotlib.ticker import FixedLocator
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
    ITERATION_LIMIT = 25
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False

    SIZES = [
        (64, 4, 0.4e6),
        (64, 5, 0.8e6),
        (64, 6, 2e6),
        (64, 7, 5e6),
        (64, 8, 30e6),
    ]

    SAMPLES_PER_POINT = {
        4: 20,
        5: 20,
        6: 20,
        7: 10,
        8: 10,
    }

    LAYERS = {
        4: [[2**4, 2**4, 2**4]],
        5: [[2**5, 2**5, 2**5]],
        6: [[2**6, 2**6, 2**6]],
        7: [[2**7, 2**7, 2**7]],
        8: [[2**8, 2**8, 2**8]],
    }

    PLOT_ESTIMATORS = ['mean']

    LOSSES = {
        4: ['squared_hinge', 'log_loss'],
        5: ['squared_hinge', 'log_loss'],
        6: ['squared_hinge', 'log_loss'],
        7: ['squared_hinge', 'log_loss'],
        8: ['squared_hinge', 'log_loss'],
    }

    METRICS = {
        4: [(-1, -1), (-1, 0), (0, -1), (0, 0)],
        5: [(-1, -1), (-1, 0), (0, -1), (0, 0)],
        6: [(-1, -1), (-1, 0), (0, -1), (0, 0)],
        7: [(-1, -1), (-1, 0), (0, -1), (0, 0)],
        8: [(-1, -1), (-1, 0), (0, -1), (0, 0)],
    }

    PATIENCES = {
        4: [2, 3, 4],
        5: [2, 3, 4],
        6: [2, 3, 4],
        7: [2, 3, 4],
        8: [2, 3, 4],
    }

    TOLERANCES = {
        4: [0.01, 0.02],
        5: [0.01, 0.02],
        6: [0.01, 0.02],
        7: [0.01, 0.02],
        8: [0.01, 0.02],
    }

    LEARNING_RATES = {
        4: [0.0005, 0.0015, 0.0025, 0.0035],
        5: [0.0005, 0.0015, 0.0025, 0.0035],
        6: [0.0005, 0.0015, 0.0025, 0.0035],
        7: [0.0005, 0.0025, 0.0045, 0.0065],
        8: [0.00025, 0.00075, 0.00125, 0.00175],
    }

    PENALTIES = {
        4: [0.0002],
        5: [0.0002],
        6: [0.0002],
        7: [0.0002],
        8: [0.0002],
    }

    BETAS_1 = {
        4: [0.9],
        5: [0.9],
        6: [0.9],
        7: [0.9],
        8: [0.9],
    }

    BETAS_2 = {
        4: [0.999],
        5: [0.999],
        6: [0.999],
        7: [0.999],
        8: [0.999],
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
            for metric_in, metric_out in self.METRICS[k]:
                validation_frac = max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N
                for layers in self.LAYERS[k]:
                    for i in range(self.SAMPLES_PER_POINT[k]):
                        for j, learning_rate in enumerate(self.LEARNING_RATES[k]):
                            cycle = i * (len(self.LEARNING_RATES[k])) + j
                            for penalty in self.PENALTIES[k]:
                                for beta_1 in self.BETAS_1[k]:
                                    for beta_2 in self.BETAS_2[k]:
                                        for patience in self.PATIENCES:
                                            for tolerance in self.TOLERANCES[k]:
                                                if metric_out == 0:
                                                    experiments.append(
                                                        ExperimentMLPScikitLearn(
                                                            progress_log_prefix=None,
                                                            parameters=Parameters_skl(
                                                                seed_simulation=0x3 + cycle,
                                                                seed_challenges=0x1415 + cycle,
                                                                seed_model=0x9265 + cycle,
                                                                seed_distance=0x3589 + cycle,
                                                                n=n,
                                                                k=k,
                                                                N=int(N),
                                                                validation_frac=validation_frac,
                                                                transformation=self.TRANSFORMATION,
                                                                combiner=self.COMBINER,
                                                                preprocessing=self.PREPROCESSING,
                                                                layers=layers,
                                                                activation=self.ACTIVATION,
                                                                metric_in=metric_in,
                                                                learning_rate=learning_rate,
                                                                penalty=penalty,
                                                                beta_1=beta_1,
                                                                beta_2=beta_2,
                                                                tolerance=tolerance,
                                                                patience=patience,
                                                                iteration_limit=self.ITERATION_LIMIT,
                                                                batch_size=1000 if k < 6 else 10000,
                                                                print_learning=self.PRINT_LEARNING,
                                                            )
                                                        )
                                                    )
                                                for loss in self.LOSSES[k]:
                                                    experiments.append(
                                                        ExperimentMLPTensorflow(
                                                            progress_log_prefix=None,
                                                            parameters=Parameters_tf(
                                                                seed_simulation=0x3 + cycle,
                                                                seed_challenges=0x1415 + cycle,
                                                                seed_model=0x9265 + cycle,
                                                                seed_distance=0x3589 + cycle,
                                                                n=n,
                                                                k=k,
                                                                N=int(N),
                                                                validation_frac=validation_frac,
                                                                transformation=self.TRANSFORMATION,
                                                                combiner=self.COMBINER,
                                                                preprocessing=self.PREPROCESSING,
                                                                layers=layers,
                                                                activation=self.ACTIVATION,
                                                                loss=loss,
                                                                metric_in=metric_in,
                                                                metric_out=metric_out,
                                                                learning_rate=learning_rate,
                                                                penalty=penalty,
                                                                beta_1=beta_1,
                                                                beta_2=beta_2,
                                                                tolerance=tolerance,
                                                                patience=patience,
                                                                iteration_limit=self.ITERATION_LIMIT,
                                                                batch_size=1000 if k < 6 else 10000,
                                                                print_learning=self.PRINT_LEARNING,
                                                            )
                                                        )
                                                    )
        return experiments

    def plot(self):
        self.plot_helper(
            name='ScikitLearn',
            df=self.experimenter.results,
        )
        self.plot_helper(
            name='Tensorflow',
            df=self.experimenter.results,
        )
        self.plot_history(
            experiment='ScikitLearn',
            name='Loss_Curves',
            df=self.experimenter.results,
            kind='loss',
        )
        self.plot_history(
            experiment='ScikitLearn',
            name='Accuracy_Curves',
            df=self.experimenter.results,
            kind='accuracy',
        )
        self.plot_history(
            experiment='Tensorflow',
            name='Loss_Curves',
            df=self.experimenter.results,
            kind='loss',
        )
        self.plot_history(
            experiment='Tensorflow',
            name='Accuracy_Curves',
            df=self.experimenter.results,
            kind='accuracy',
        )

    def plot_helper(self, name, df):
        df = df[df['experiment'] == 'ExperimentMLP' + name]
        ks = sorted(list(set(self.experimenter.results['k'])))
        ncols = len(ks)
        nrows = len(self.PLOT_ESTIMATORS) + 2
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(8*ncols, 3 * nrows)

        axes = axes.reshape((nrows, ncols))

        for j, k in enumerate(ks):
            data = df[df['k'] == k]

            for i, estimator in enumerate(self.PLOT_ESTIMATORS):
                lineplot(
                    x='learning_rate',
                    y='accuracy',
                    hue='metric_in',
                    style='metric_out',
                    data=data,
                    ax=axes[i][j],
                    legend=False,
                    estimator=get_estimator(estimator),
                    ci=None,
                )
                scatterplot(
                    x='learning_rate',
                    y='accuracy',
                    hue='metric_in',
                    style='metric_out',
                    data=data,
                    ax=axes[i][j],
                    legend='full' if i == 0 else False,
                )
                axes[0][j].legend(loc='upper right', bbox_to_anchor=(1.28, 1.1))
                axes[0][j].set_title('k={},   {} experiments per combination,   {}/{}\n'.format(
                    k, self.SAMPLES_PER_POINT[k], len(data),
                    len(self.experimenter.results[(self.experimenter.results['k'] == k) & (
                        self.experimenter.results['experiment'] == 'ExperimentMLP' + name)])))
                axes[i][j].set_ylabel('accuracy {}'.format(estimator))
                axes[i][j].set_ylim((-0.05 if estimator.startswith('success') else 0.45, 1.05))
                axes[i][j].xaxis.set_major_locator(FixedLocator(list(set(
                    self.experimenter.results['learning_rate'][self.experimenter.results['k'] == k]))))

            lineplot(
                x='learning_rate',
                y='accuracy',
                hue='metric_in',
                style='metric_out',
                data=data,
                ax=axes[-2][j],
                legend=False,
                estimator=get_estimator('mean'),
                ci=None,
            )
            scatterplot(
                x='learning_rate',
                y='accuracy',
                hue='metric_in',
                style='metric_out',
                data=data,
                ax=axes[-2][j],
                legend=False,
            )

            axes[-2][j].xaxis.set_major_locator(FixedLocator(list(set(
                self.experimenter.results['learning_rate'][self.experimenter.results['k'] == k]))))
            axes[-2][j].yaxis.set_minor_locator(FixedLocator([.7, .9, .98]))
            axes[-2][j].grid(b=True, which='minor', color='gray', linestyle='--')

            lineplot(
                x='accuracy',
                y='measured_time',
                hue='metric_in',
                style='metric_out',
                data=data,
                ax=axes[-1][j],
                estimator='mean',
                ci=None,
            )
            y = self.REFERENCE_TIMES[(64, k)]
            axes[-1][j].plot((0.5, 1), (y, y), label='reference: {}'.format(y))
            axes[-1][j].set_xscale('linear')
            axes[-1][j].set_yscale('linear')
            axes[-1][j].set_xlabel('accuracy')
            axes[-1][j].set_ylabel('runtime in s')
            axes[-1][j].set_ylim(bottom=0)
            axes[-1][j].xaxis.set_minor_locator(FixedLocator([.7, .9, .98]))
            axes[-1][j].grid(b=True, which='minor', color='gray', linestyle='--')
            axes[-1][j].legend(loc='upper right', bbox_to_anchor=(1.28, 1.1))
            for i in range(nrows - 1):
                ticks = axes[i][j].get_xticks()
                e = int(log10(min(float(tick) for tick in ticks))) - 1
                axes[i][j].set_xticklabels(['{0:.1f}'.format(float(tick) * 10**(-e)) for tick in ticks])
                axes[i][j].set_xticklabels([str(round(float(label))) if float(label).is_integer() else label
                                            for label in [item.get_text() for item in axes[i][j].get_xticklabels()]])
                axes[i][j].set_xlabel('learning_rate times 1e{}'.format(e))
        fig.subplots_adjust(hspace=.5, wspace=.5)
        title = fig.suptitle('Accuracy of {} Results on XOR Arbiter PUF'.format(name))
        title.set_position([.5, 1.05])

        fig.savefig('figures/{}_{}.pdf'.format(self.name(), name), bbox_inches='tight', pad_inches=.5)

    def plot_history(self, experiment, name, df, kind):
        df = df[df['experiment'] == 'ExperimentMLP' + experiment]
        ks = sorted(list(set(self.experimenter.results['k'])))
        intervals = {(.0, .7): 'bad', (.7, .9): 'medium', (.9, .98): 'good', (.98, 1.): 'perfect'}
        ncols = len(ks)
        nrows = len(intervals)
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(8 * ncols, 3 * nrows)
        axes = axes.reshape((nrows, ncols))

        for j, k in enumerate(ks):
            data = df[df['k'] == k]
            axes[0][j].set_title('k={},   {} experiments per combination,   {}/{}\n'.format(
                k, self.SAMPLES_PER_POINT[k], len(data),
                len(self.experimenter.results[(self.experimenter.results['k'] == k) & (
                        self.experimenter.results['experiment'] == 'ExperimentMLP' + experiment)])))
            for i, (low, high) in enumerate(intervals):
                curves = data[(data.accuracy >= low) & (data.accuracy < high)][kind + '_curve']
                axes[i][j].set_ylabel('{} results\n{}'.format(intervals[(low, high)], kind))
                axes[i][j].set_xlabel('{}'.format('epoch'))
                for curve in curves:
                    axes[i][j].plot([float(s) for s in findall(pattern=r'-?\d+\.?\d*', string=curve)])

        fig.subplots_adjust(hspace=.5, wspace=.5)
        title = fig.suptitle('History of {} of {} Results on XOR Arbiter PUFs'.format(kind, experiment))
        title.set_position([.5, 1.05])

        fig.savefig('figures/{}_{}_{}.pdf'.format(self.name(), name, experiment), bbox_inches='tight', pad_inches=.5)
