from math import sqrt

from matplotlib.pyplot import subplots
from matplotlib.ticker import FixedLocator, MultipleLocator, FuncFormatter, FormatStrFormatter
from numpy import linspace
from seaborn import lineplot, scatterplot

from pypuf.studies.base import Study
from pypuf.experiments.experiment.mlp import ExperimentMLP, Parameters


class AseeriEtAlStudy(Study):

    CONFIGURATIONS = [
        #  n, k,    N,  learning_rate
        ( 64, 4,   .4, .01),
        ( 64, 5,   .8, .005),
        ( 64, 6,  2,   .007),
        ( 64, 7,  5,   .0045),
        ( 64, 8, 30,   .00175),
        #(128, 4,  .8),
        #(128, 5,  3),
        #(128, 6, 20),
        #(128, 7, 40),
    ]

    PLOT_ESTIMATORS = ['min', 'mean', 'max']

    TOLS = [0.1]  # [0.001, 0.025, 0.05, 0.075, 0.1]
    LEARNING_RATES = [0.01]  # linspace(.2e-2, 1.8e-2, 9)

    def experiments(self):
        e = []
        for i in range(100):
            for (n, k, N, learning_rate) in self.CONFIGURATIONS:
                for tol in self.TOLS:
                    e.append(
                        ExperimentMLP(
                            log_name=None,
                            parameters=Parameters(
                                n=n,
                                k=k,
                                N=int(N * 1e6),
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
                                n_iter_no_change=10,
                            )
                        )
                    )
        return e

    def plot(self):
        fig, axes = subplots(nrows=len(self.PLOT_ESTIMATORS) + 2, ncols=1)
        fig.set_size_inches(8, 6 * len(self.PLOT_ESTIMATORS))

        if len(self.PLOT_ESTIMATORS) == 1:
            axes = [axes]

        for i, estimator in enumerate(self.PLOT_ESTIMATORS):
            lineplot(
                x='k',
                y='accuracy',
                style='n',
                data=self.experimenter.results,
                ax=axes[i],
                legend='full',
                estimator=estimator,
                ci=None,
            )
            axes[i].set_ylabel('accuracy %s' % estimator)
            axes[i].xaxis.set_major_locator(MultipleLocator(1))
            axes[i].set_ylim(.45, 1.)

        scatterplot(
            x='k',
            y='accuracy',
            style='n',
            data=self.experimenter.results,
            ax=axes[-2],
            legend='full',
        )
        axes[-2].xaxis.set_major_locator(MultipleLocator(1))

        data = self.experimenter.results
        data['distance'] = 1 - data['accuracy']

        lineplot(
            x='distance',
            y='measured_time',
            style='n',
            hue='k',
            data=data,
            ax=axes[-1],
            legend='full',
            estimator='mean',
        )

        axes[-1].set_xscale('log')
        axes[-1].set_yscale('log')
        axes[-1].xaxis.set_major_locator(MultipleLocator(.2))
        axes[-1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axes[-1].yaxis.set_major_locator(FixedLocator([
            # References for:
            19.2,  # k = 4
            58,  # k = 5
            7.4 * 60,  # k = 6
            11.8 * 60,  # k = 7
            23.3 * 60,  # k = 8
        ]))
        axes[-1].yaxis.set_major_formatter(FuncFormatter(
            lambda x, _: '%3.1fs' % x if x < 60 else '%2.1fmin' % (x / 60)))

        fig.subplots_adjust(hspace=.5, wspace=.5)
        fig.suptitle('Accuracy of Scikit-Learn Results on XOR Arbiter PUF')

        fig.savefig('figures/%s.pdf' % (self.name()), bbox_inches='tight', pad_inches=.5)
