from matplotlib.pyplot import subplots
from matplotlib.ticker import FixedLocator, MultipleLocator, FuncFormatter, FormatStrFormatter
from seaborn import lineplot, scatterplot

from pypuf.studies.base import Study
from pypuf.experiments.experiment.mlp import ExperimentMLP, Parameters


class AseeriEtAlHyperparameterStudy(Study):

    SIZES = [
        (64, 4, 0.4e6),
        (64, 5, 0.8e6),
        (64, 6, 2e6),
        (64, 7, 5e6),
        (64, 8, 20e6),
    ]

    PLOT_ESTIMATORS = ['min', 'mean', 'max']

    TOLS = {
        4: [0.1, 0.2, 0.5],
        5: [0.1, 0.2, 0.5],
        6: [0.1, 0.2, 0.5],
        7: [0.05, 0.1, 0.2, 0.5, .75, 1.0],
        8: [0.1, 0.2, 0.5, .75, 1.0],
    }
    LEARNING_RATES = {
        4: [.2e-2, .4e-2, .5e-2, .6e-2, .7e-2, .8e-2, 1e-2],
        5: [.2e-2, .4e-2, .5e-2, .6e-2, .7e-2, .8e-2, 1e-2],
        6: [.2e-2, .4e-2, .5e-2, .6e-2, .7e-2, .8e-2, 1e-2],
        7: [.2e-2, .4e-2, .425e-2, .450e-2, .475e-2, .5e-2, .6e-2, .7e-2, .8e-2, 1e-2],
        8: [0.01e-1, 0.02e-2, .05e-2, .1e-2, .125e-2, .150e-2, .175e-2, .2e-2, .4e-2, .5e-2, .6e-2, .7e-2, .8e-2, 1e-2],
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
        e = []
        for i in range(10):
            for (n, k, N) in self.SIZES:
                for tol in self.TOLS[k]:
                    for learning_rate in self.LEARNING_RATES[k]:
                        e.append(
                            ExperimentMLP(
                                log_name=None,
                                parameters=Parameters(
                                    n=n,
                                    k=k,
                                    N=int(N),
                                    seed_instance=0x1 + i,
                                    seed_model=0x1000 + i,
                                    transformation='id',
                                    combiner='xor',
                                    seed_distance=0x2 + i,
                                    seed_accuracy=0x3 + i,
                                    batch_size=1000 if k < 6 else 10000,
                                    iteration_limit=100,
                                    initial_model_sigma=1,
                                    learning_rate=learning_rate,
                                    activation='relu',
                                    hidden_layer_sizes=(2**k, 2**k, 2**k),
                                    tol=tol,
                                    n_iter_no_change=5,
                                )
                            )
                        )
        return e

    def plot(self):
        ks = sorted(list(set(self.experimenter.results['k'])))

        format_time = lambda x, _ = None: '%.1fs' % x if x < 60 else '%.1fmin' % (x / 60)

        fig, axes = subplots(ncols=len(ks), nrows=len(self.PLOT_ESTIMATORS) + 2)
        fig.set_size_inches(8*len(ks), 6 * len(self.PLOT_ESTIMATORS))

        axes = axes.reshape((len(self.PLOT_ESTIMATORS) + 2, len(ks)))

        self.experimenter.results['distance'] = 1 - self.experimenter.results['accuracy']
        for j, k in enumerate(ks):
            data = self.experimenter.results[self.experimenter.results['k'] == k]

            for i, estimator in enumerate(self.PLOT_ESTIMATORS):
                lineplot(
                    x='learning_rate',
                    y='accuracy',
                    style='tol',
                    data=data,
                    ax=axes[i][j],
                    legend='full',
                    estimator=estimator,
                    ci=None,
                )
                scatterplot(
                    x='learning_rate',
                    y='accuracy',
                    style='tol',
                    data=data,
                    ax=axes[i][j],
                    legend='full',
                )
                if estimator == 'mean':
                    lineplot(
                        x=self.LEARNING_RATES[k],
                        y=[self.REFERENCE_MEAN_ACCURACY[(64, k)]] * len(self.LEARNING_RATES[k]),
                        ax=axes[i][j],
                        label='reference',
                        legend='full',
                    )
                axes[i][j].set_ylabel('k=%i\naccuracy %s' % (k, estimator))
                axes[i][j].set_ylim((.45, 1.05))

            scatterplot(
                x='learning_rate',
                y='accuracy',
                style='tol',
                data=data,
                ax=axes[-2][j],
                legend='full',
            )

            lineplot(
                x='distance',
                y='measured_time',
                style='tol',
                hue='learning_rate',
                data=data,
                ax=axes[-1][j],
                legend='brief',
                estimator='mean',
            )

            lineplot(
                x=[0, .5],
                y=[self.REFERENCE_TIMES[(64, k)]] * 2,
                ax=axes[-1][j],
                label='reference',
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
        fig.suptitle('Accuracy of Scikit-Learn Results on XOR Arbiter PUF')

        fig.savefig('figures/%s.pdf' % (self.name()), bbox_inches='tight', pad_inches=.5)
