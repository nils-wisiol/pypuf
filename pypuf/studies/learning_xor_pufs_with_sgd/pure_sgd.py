from itertools import product
from random import seed

from matplotlib.pyplot import subplots
from matplotlib.ticker import FixedLocator
from numpy import ones
from numpy.ma import floor, log10
from seaborn import color_palette, lineplot, scatterplot, stripplot

from pypuf.studies.base import Study
from pypuf.experiments.experiment.pure_sgd import ExperimentPureStochasticGradientDescent, Parameters


class PureStochasticGradientDescentStudy(Study):
    TRANSFORMATION = 'id'
    COMBINER = 'xor'
    ITERATION_LIMIT = 100
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False

    PLOT_ESTIMATORS = []

    SIZES = [
        (64, 1, 0.05e6),
        (64, 2, 0.1e6),
        (64, 3, 0.2e6),
        (64, 4, 0.4e6),
        # (64, 5, 0.8e6),
        # (64, 6, 2e6),
        # (64, 7, 5e6),
        # (64, 8, 30e6),
    ]

    SAMPLES_PER_POINT = {
        1: 3,
        2: 3,
        3: 3,
        4: 3,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
    }

    LOSSES = {
        1: ['log_loss'],
        2: ['log_loss'],
        3: ['log_loss'],
        4: ['log_loss'],
        5: ['log_loss'],
        6: ['log_loss'],
        7: ['log_loss'],
        8: ['log_loss'],
    }

    PATIENCE = {
        1: [5],
        2: [5],
        3: [5],
        4: [5],
        5: [5],
        6: [5],
        7: [5],
        8: [5],
    }

    TOLERANCES = {
        1: [0.0025],
        2: [0.0025],
        3: [0.0025],
        4: [0.0025],
        5: [0.0025],
        6: [0.0025],
        7: [0.0025],
        8: [0.0025],
    }

    LEARNING_RATES = {
        1: [0.0001, 0.001, 0.01, 0.1],
        2: [0.0001, 0.001, 0.01, 0.1],
        3: [0.0001, 0.001, 0.01, 0.1],
        4: [0.0001, 0.001, 0.01, 0.1],
        5: [0.01],
        6: [0.01],
        7: [0.01],
        8: [0.01],
    }

    PENALTIES = {
        1: [0.0002, 0.001, 0.01],
        2: [0.0002, 0.001, 0.01],
        3: [0.0002, 0.001, 0.01],
        4: [0.0002, 0.001, 0.01],
        5: [0.0002, 0.001, 0.01],
        6: [0.0002, 0.001, 0.01],
        7: [0.0002, 0.001, 0.01],
        8: [0.0002, 0.001, 0.01],
    }

    BETAS_1 = {
        1: [0.9],
        2: [0.9],
        3: [0.9],
        4: [0.9],
        5: [0.9],
        6: [0.9],
        7: [0.9],
        8: [0.9],
    }

    BETAS_2 = {
        1: [0.999],
        2: [0.999],
        3: [0.999],
        4: [0.999],
        5: [0.999],
        6: [0.999],
        7: [0.999],
        8: [0.999],
    }

    REFERENCE_TIMES = {
        (64, 1): 3,
        (64, 2): 7,
        (64, 3): 13,
        (64, 4): 19.2,
        (64, 5): 58,
        (64, 6): 7.4 * 60,
        (64, 7): 11.8 * 60,
        (64, 8): 23.3 * 60,
    }

    REFERENCE_MEAN_ACCURACY = {
        (64, 1): .9842,
        (64, 2): .9842,
        (64, 3): .9842,
        (64, 4): .9842,
        (64, 5): .9855,
        (64, 6): .9915,
        (64, 7): .9921,
        (64, 8): .9874,
    }

    EXPERIMENTS = []

    def experiments(self):
        """
        Generate an experiment for every parameter combination corresponding to the definitions above.
        For each combination of (size, samples_per_point, learning_rate) different random seeds are used.
        """
        for c1, (n, k, N) in enumerate(self.SIZES):
            for c2 in range(self.SAMPLES_PER_POINT[k]):
                for c3, learning_rate in enumerate(self.LEARNING_RATES[k]):
                    cycle = c1 * (self.SAMPLES_PER_POINT[k] * len(self.LEARNING_RATES[k])) \
                            + c2 * len(self.LEARNING_RATES[k]) + c3
                    validation_frac = max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N
                    for (penalty, beta_1, beta_2, patience, tolerance, loss) in list(product(
                            *[self.PENALTIES[k], self.BETAS_1[k], self.BETAS_2[k], self.PATIENCE[k],
                              self.TOLERANCES[k], self.LOSSES[k]]
                    )):
                        self.EXPERIMENTS.append(
                            ExperimentPureStochasticGradientDescent(
                                progress_log_prefix=None,
                                parameters=Parameters(
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
                                    loss=loss,
                                    learning_rate=learning_rate,
                                    penalty=penalty,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    tolerance=tolerance,
                                    patience=patience,
                                    iteration_limit=self.ITERATION_LIMIT,
                                    batch_size=1,    # BUG!!!   1000 if k < 4 else 1000 if k < 6 else 10000,
                                    print_learning=self.PRINT_LEARNING,
                                )
                            )
                        )
        return self.EXPERIMENTS

    def plot(self):
        if not self.EXPERIMENTS:
            self.experiments()
        self.plot_helper(
            df=self.experimenter.results,
            param_x='learning_rate',
            param_1='N',
            param_2=None,
        )

    def plot_helper(self, df, param_x, param_1, param_2=None):
        name = 'ExperimentPureStochasticGradientDescent'
        param_y = 'accuracy'
        ncols = len(self.SIZES)
        nrows = 2
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(9 * ncols, 4 * nrows)
        axes = axes.reshape((nrows, ncols))
        alpha_scatter = 0.7
        alpha_lines = 0.7
        palette = color_palette('plasma')
        for j, (n, k, N) in enumerate(self.SIZES):
            data = df[(df['k'] == k) & (df['n'] == n)]
            lineplot(
                x=param_x,
                y=param_y,
                hue=param_1,
                style=param_2,
                data=data,
                ax=axes[0][j],
                legend=False,
                estimator='mean',
                ci=None,
                alpha=alpha_lines,
                #palette=palette,
            )
            scatterplot(
                x=param_x,
                y=param_y,
                hue=param_1,
                style=param_2,
                data=data,
                ax=axes[0][j],
                legend='full',
                alpha=alpha_scatter,
                #palette=palette,
            )
            total = sum([e.parameters.k == k and e.parameters.n == n for e in self.EXPERIMENTS])
            axes[0][j].set_title('n={}, k={}\n\n{} experiments per combination,   {}/{}\n'.format(
                n, k, self.SAMPLES_PER_POINT[(k)], len(data), total))
            axes[0][j].legend(loc='upper right', bbox_to_anchor=(1.235, 1.03))
            lineplot(
                x='accuracy',
                y='measured_time',
                hue=param_1,
                style=param_2,
                data=data,
                ax=axes[1][j],
                legend='full',
                estimator='mean',
                ci=None,
                alpha=alpha_lines,
                #palette=palette,
            )
            axes[1][j].set_xscale('linear')
            axes[1][j].set_yscale('linear')
            axes[1][j].set_xlabel(param_y)
            axes[1][j].set_ylabel('runtime in s')
            axes[1][j].set_ylim(bottom=0)
            axes[1][j].xaxis.set_minor_locator(FixedLocator([.7, .9, .98]))
            axes[1][j].grid(b=True, which='minor', color='gray', linestyle=':')
            axes[1][j].legend(loc='upper right', bbox_to_anchor=(1.235, 1.03))
        fig.subplots_adjust(hspace=.5, wspace=.5)
        title = fig.suptitle('Accuracy of {} Results on XOR Arbiter PUF'.format(name))
        title.set_position([.5, 1.05])
        fig.savefig('figures/{}_{}.pdf'.format(self.name(), name), bbox_inches='tight', dpi=300, pad_inches=.5)

        width = 0.3
        seed(42)
        df = self.experimenter.results
        distances = 1 - df.accuracy
        df['distance'] = distances
        ncols = len(set([k for n, k, N in self.SIZES]))
        nrows = 3
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(7 * ncols, 4 * nrows)
        axes = axes.reshape((nrows, ncols))
        marker = 'o'

        for i, k in enumerate(list(sorted(set(df.k)))):
            stripplot(
                x='n',
                y='accuracy',
                data=df[df.k == k],
                ax=axes[0][i],
                jitter=True,
                zorder=1,
                marker=marker,
                alpha=alpha_scatter,
                #palette=palette,
            )
            for j, n in enumerate(list(sorted(set(df.n)))):
                Ns = list(sorted(set(df[(df.k == k) & (df.n == n)].N)))
                for l, N in enumerate(Ns):
                    mean = df[(df.k == k) & (df.n == n) & (df.N == N)].accuracy.mean()
                    axes[0][i].plot([j - width / 2, j + width / 2], [mean, mean],
                                    color=(l / len(Ns),) * 3, linewidth=1, zorder=2, label=l if j == 0 else None)
            axes[0][i].set_title('k={}\n'.format(k))
            axes[0][i].set_yscale('linear')
            axes[0][i].set_ylabel('accuracy')

            stripplot(
                x='n',
                y='distance',
                data=df[df.k == k],
                ax=axes[1][i],
                jitter=True,
                zorder=1,
                marker=marker,
                alpha=alpha_scatter,
                #palette=palette,
            )
            for j, n in enumerate(list(sorted(set(df.n)))):
                Ns = list(sorted(set(df[(df.k == k) & (df.n == n)].N)))
                for l, N in enumerate(Ns):
                    mean = df[(df.k == k) & (df.n == n) & (df.N == N)].distance.mean()
                    axes[1][i].plot([j - width / 2, j + width / 2], [mean, mean],
                                    color=(l / len(Ns),) * 3, linewidth=1, zorder=2, label=l if j == 0 else None)
            axes[1][i].set_yscale('log')
            axes[1][i].invert_yaxis()
            axes[1][i].set_ylim(top=0.002, bottom=1.0)
            major_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
            minor_ticks = [0.01, 0.02, 0.03, 0.04, 0.05]
            axes[1][i].set_yticks(ticks=major_ticks, minor=False)
            axes[1][i].set_yticks(ticks=minor_ticks, minor=True)
            axes[1][i].set_yticklabels(ones(shape=5) - major_ticks, minor=False)
            axes[1][i].set_yticklabels(ones(shape=5) - minor_ticks, minor=True)
            axes[1][i].grid(b=True, which='minor', color='gray', linestyle='--')
            axes[1][i].set_ylabel('accuracy')

            stripplot(
                x='n',
                y='measured_time',
                data=df[df.k == k],
                ax=axes[2][i],
                jitter=True,
                zorder=1,
                marker=marker,
                alpha=alpha_scatter,
                #palette=palette,
            )
            for j, n in enumerate(list(sorted(set(df.n)))):
                Ns = list(sorted(set(df[(df.k == k) & (df.n == n)].N)))
                for l, N in enumerate(Ns):
                    mean = df[(df.k == k) & (df.n == n) & (df.N == N)].measured_time.mean()
                    axes[2][i].plot([j - width / 2, j + width / 2], [mean, mean],
                                    color=(l / len(Ns),) * 3, linewidth=1, zorder=2, label=l if j == 0 else None)
            axes[2][i].set_yscale('log')

        axes[2][0].set_ylabel('runtime in s')
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        title = fig.suptitle('Overview of Learning Results on XOR Arbiter PUFs of length 64\n'
                             'using a Pure Stochastic Gradient Descent on PUF simulation implemented in Tensorflow.',
                             size=16)
        title.set_position([0.5, 1.0])
        fig.savefig('figures/{}_overview.pdf'.format(self.name()), bbox_inches='tight', dpi=300, pad_inches=.5)
