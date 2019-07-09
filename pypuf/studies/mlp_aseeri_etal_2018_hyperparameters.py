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
    ITERATION_LIMIT = 40
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False

    PLOT_ESTIMATORS = []

    SIZES = [
        (64, 4, 0.4e6),
        (64, 5, 0.8e6),
        (64, 6, 2e6),
        (64, 7, 5e6),
        (64, 8, 30e6),
    ]

    SAMPLES_PER_POINT = {
        4: 10,
        5: 10,
        6: 10,
        7: 10,
        8: 10,
    }

    LAYERS = {
        4: [[2 ** 4, 2 ** 4, 2 ** 4]],
        5: [[2 ** 5, 2 ** 5, 2 ** 5]],
        6: [[2 ** 6, 2 ** 6, 2 ** 6]],
        7: [[2 ** 7, 2 ** 7, 2 ** 7]],
        8: [[2 ** 8, 2 ** 8, 2 ** 8]],
    }

    LOSSES = {
        4: ['log_loss', 'squared_hinge'],
        5: ['log_loss', 'squared_hinge'],
        6: ['log_loss', 'squared_hinge'],
        7: ['log_loss', 'squared_hinge'],
        8: ['log_loss', 'squared_hinge'],
    }

    DOMAINS = {
        4: [(-1, -1)],
        5: [(-1, -1)],
        6: [(-1, -1)],
        7: [(-1, -1)],
        8: [(-1, -1)],
    }

    PATIENCE = {
        4: [4],
        5: [4],
        6: [4],
        7: [4],
        8: [4],
    }

    TOLERANCES = {
        4: [0.0025],
        5: [0.0025],
        6: [0.0025],
        7: [0.0025],
        8: [0.0025],
    }

    LEARNING_RATES = {
        4: [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007],
        5: [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007],
        6: [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007],
        7: [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007],
        8: [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007],
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
        (128, 7): 1.5 * 60 ** 2,
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

    EXPERIMENTS = []

    def experiments(self):
        for (n, k, N) in self.SIZES:
            for domain_in, domain_out in self.DOMAINS[k]:
                validation_frac = max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N
                for layers in self.LAYERS[k]:
                    for i in range(self.SAMPLES_PER_POINT[k]):
                        for j, learning_rate in enumerate(self.LEARNING_RATES[k]):
                            cycle = i * (len(self.LEARNING_RATES[k])) + j
                            for penalty in self.PENALTIES[k]:
                                for beta_1 in self.BETAS_1[k]:
                                    for beta_2 in self.BETAS_2[k]:
                                        for patience in self.PATIENCE[k]:
                                            for tolerance in self.TOLERANCES[k]:
                                                # if domain_out == 0:
                                                self.EXPERIMENTS.append(
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
                                                            domain_in=domain_in,
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
                                                    self.EXPERIMENTS.append(
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
                                                                domain_in=domain_in,
                                                                domain_out=domain_out,
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
        return self.EXPERIMENTS

    def plot(self):
        if not self.EXPERIMENTS:
            self.experiments()
        self.plot_helper(
            name='ScikitLearn',
            df=self.experimenter.results,
            param_x='learning_rate',
            param_1=None,
            param_2=None,
        )
        self.plot_helper(
            name='Tensorflow',
            df=self.experimenter.results,
            param_x='learning_rate',
            param_1='loss',
            param_2=None,
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

    def plot_helper(self, name, df, param_x, param_1, param_2=None):
        param_y = 'accuracy'
        df = df[df['experiment'] == 'ExperimentMLP' + name]
        ks = sorted(list(set(self.experimenter.results['k'])))
        ncols = len(ks)
        num_estimators = len(self.PLOT_ESTIMATORS)
        nrows = num_estimators + 3 if name == 'Tensorflow' else num_estimators + 2
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(9*ncols, 4 * nrows)

        axes = axes.reshape((nrows, ncols))

        for j, k in enumerate(ks):
            data = df[df['k'] == k]

            for i, estimator in enumerate(self.PLOT_ESTIMATORS):
                lineplot(
                    x=param_x,
                    y=param_y,
                    hue=param_1,
                    style=param_2,
                    data=data,
                    ax=axes[i][j],
                    legend=False,
                    estimator=get_estimator(estimator),
                    ci=None,
                )
                scatterplot(
                    x=param_x,
                    y=param_y,
                    hue=param_1,
                    style=param_2,
                    data=data,
                    ax=axes[i][j],
                    legend='full' if i == 0 else False,
                )
                axes[i][j].set_ylabel('accuracy {}'.format(estimator))
                axes[i][j].set_ylim((-0.05 if estimator.startswith('success') else 0.45, 1.05))
                axes[i][j].xaxis.set_major_locator(FixedLocator(list(set(
                    self.experimenter.results[param_x][self.experimenter.results['k'] == k]))))

            lineplot(
                x=param_x,
                y=param_y,
                hue=param_1,
                style=param_2,
                data=data,
                ax=axes[num_estimators][j],
                legend=False,
                estimator=get_estimator('mean'),
                ci=None,
            )
            scatterplot(
                x=param_x,
                y=param_y,
                hue=param_1,
                style=param_2,
                data=data,
                ax=axes[num_estimators][j],
                legend='full',
            )
            if param_1:
                axes[0][j].legend(loc='upper right', bbox_to_anchor=(1.25, 1.06))
            lib = 'tensorflow' if name == 'Tensorflow' else 'scikit-learn' if name == 'ScikitLearn' else ''
            total = sum([e.parameters.k == k and lib in e.NAME for e in self.EXPERIMENTS])
            axes[0][j].set_title('k={}\n\n{} experiments per combination,   {}/{}\n'.format(
                k, self.SAMPLES_PER_POINT[k], len(data), total))

            axes[num_estimators][j].xaxis.set_major_locator(FixedLocator(list(set(
                self.experimenter.results[param_x][self.experimenter.results['k'] == k]))))
            axes[num_estimators][j].yaxis.set_minor_locator(FixedLocator([.7, .9, .98]))
            axes[num_estimators][j].grid(b=True, which='minor', color='gray', linestyle=':')

            if name == 'Tensorflow':
                lineplot(
                    x=param_x,
                    y=param_y,
                    hue='loss',
                    data=data,
                    ax=axes[-2][j],
                    legend=False,
                    estimator=get_estimator('mean'),
                    ci=None,
                )
                scatterplot(
                    x=param_x,
                    y=param_y,
                    hue='loss',
                    data=data,
                    ax=axes[-2][j],
                    legend='full',
                )

            axes[-2][j].xaxis.set_major_locator(FixedLocator(list(set(
                self.experimenter.results[param_x][self.experimenter.results['k'] == k]))))
            axes[-2][j].yaxis.set_minor_locator(FixedLocator([.7, .9, .98]))
            axes[-2][j].grid(b=True, which='minor', color='gray', linestyle=':')
            if param_1:
                axes[-2][j].legend(loc='upper right', bbox_to_anchor=(1.325, 1.06))

            lineplot(
                x='accuracy',
                y='measured_time',
                hue=param_1,
                style=param_2,
                data=data,
                ax=axes[-1][j],
                legend='full',
                estimator='mean',
                ci=None,
            )
            y = self.REFERENCE_TIMES[(64, k)]
            axes[-1][j].plot((0.5, 1), (y, y), label='reference: {}'.format(y))
            axes[-1][j].set_xscale('linear')
            axes[-1][j].set_yscale('linear')
            axes[-1][j].set_xlabel(param_y)
            axes[-1][j].set_ylabel('runtime in s')
            axes[-1][j].set_ylim(bottom=0)
            axes[-1][j].xaxis.set_minor_locator(FixedLocator([.7, .9, .98]))
            axes[-1][j].grid(b=True, which='minor', color='gray', linestyle=':')
            if len([self.experimenter.results['k'] == k]) > 0:
                axes[-1][j].legend(loc='upper right', bbox_to_anchor=(1.275 + 0.015*len(str(y)), 1.06))
            for i in range(nrows - 1):
                ticks = axes[i][j].get_xticks()
                tmp = min([float(tick) for tick in ticks])
                e = int(log10(tmp if tmp > 0 else .0001)) - 1
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
        lib = 'tensorflow' if name == 'Tensorflow' else 'scikit-learn' if name == 'ScikitLearn' else ''
        ks = sorted(list(set(self.experimenter.results['k'])))
        intervals = {(.0, .7): 'bad', (.7, .9): 'medium', (.9, .98): 'good', (.98, 1.): 'perfect'}
        ncols = len(ks)
        nrows = len(intervals)
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(8 * ncols, 3 * nrows)
        axes = axes.reshape((nrows, ncols))

        for j, k in enumerate(ks):
            data = df[df['k'] == k]
            total = sum([e.parameters.k == k and lib in e.NAME for e in self.EXPERIMENTS])
            axes[0][j].set_title('k={}\n\n{} experiments per combination,   {}/{}\n'.format(
                k, self.SAMPLES_PER_POINT[k], len(data), total))
            for i, (low, high) in enumerate(intervals):
                curves = data[(data.accuracy >= low) & (data.accuracy < high)][kind + '_curve']
                axes[i][j].set_ylabel('{} results\n\n{}'.format(intervals[(low, high)], kind) if j == 0
                                      else '{}'.format(kind))
                axes[i][j].set_xlabel('{}'.format('epoch'))
                for curve in curves:
                    axes[i][j].plot([float(s) for s in findall(pattern=r'-?\d+\.?\d*', string=str(curve))])

        fig.subplots_adjust(hspace=.5, wspace=.5)
        title = fig.suptitle('History of {} of {} Results on XOR Arbiter PUFs'.format(kind, experiment))
        title.set_position([.5, 1.05])

        fig.savefig('figures/{}_{}_{}.pdf'.format(self.name(), name, experiment), bbox_inches='tight', pad_inches=.5)
