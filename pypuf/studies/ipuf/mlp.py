"""
This module describes a study that defines a set of experiments in order to examine the quality of Deep Learning based
modeling attacks on XOR Arbiter PUFs. Furthermore, some plots are defined to visualize the experiment's results.
The Deep Learning technique used here is the Feed-Forward Neural Network architecture called Multilayer Perceptron (MLP)
[1] that is applied with the optimization technique Adam [2] for a Stochastic Gradient Descent. Implementations of the
MLP and Adam are used from Scikit-Learn [3].

References:
[1]  F. Rosenblatt,         "The Perceptron: A Probabilistic Model for Information Storage and Organization in the
                            Brain.", Psychological Review, volume 65, pp. 386-408, 1958.
[2]  D. Kingma and J. Ba,   “Adam: A Method for Stochastic Optimization”, arXiv:1412.6980, 2014.
[3]  F., Pedregosa et al.,  "Scikit-learn: Machine learning in Python", Journal of Machine Learning Research, volume 12,
                            pp. 2825-2830, 2011.
                            https://scikit-learn.org
"""
import re
from matplotlib.pyplot import close, subplots
from matplotlib.ticker import FixedLocator
from numpy import isnan, ones
from numpy.random.mtrand import seed
from seaborn import catplot, axes_style, color_palette, lineplot, scatterplot, stripplot

from pypuf.experiments.experiment.mlp_skl import ExperimentMLPScikitLearn, Parameters
from pypuf.studies.base import Study


class InterposeMLPStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """
    SHUFFLE = True

    ITERATION_LIMIT = 50
    PATIENCE = ITERATION_LIMIT
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False

    SIZES = {
                (6, 6): (
                    [8 * 10**6, 16 * 10**6, 32 * 10**6, 64 * 10**6],  # up to 40GB
                    [0.0002, 0.0004, 0.0006, 0.0008, 0.001],
                ),
                (7, 7): (
                    [40 * 10**6, 80 * 10**6, 103 * 10**6],  # up to 62.5GB
                    [0.0002, 0.0004, 0.0006, 0.0008, 0.001],
                ),
                (8, 8): (
                    [40 * 10**6, 80 * 10**6, 103 * 10**6],  # up to 62.5GB
                    [0.0002, 0.0004, 0.0006, 0.0008, 0.001],
                ),
                (9, 9): (
                    [40 * 10**6, 80 * 10**6, 103 * 10**6],  # up to 62.5GB
                    [0.0002, 0.0004, 0.0006, 0.0008, 0.001],
                ),
                (1, 8): (
                    [20 * 10**6, 40 * 10**6, 80 * 10**6, 103 * 10**6],  # up to 62.5GB
                    [0.0001, 0.0008, 0.004],
                ),
                (1, 9): (
                    [40 * 10**6, 80 * 10**6, 103 * 10**6],  # up to 62.5GB
                    [0.00005, 0.0001, 0.0008, 0.004],
                ),
            }

    def experiments(self):
        return [
            ExperimentMLPScikitLearn(
                progress_log_prefix=self.name(),
                parameters=Parameters(
                    seed_simulation=0x3 + seed,
                    seed_challenges=0x1415 + seed,
                    seed_model=0x9265 + seed,
                    seed_distance=0x3589 + seed,
                    n=n,
                    k_up=k_up,
                    k_down=k_down,
                    N=N,
                    validation_frac=max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N,
                    combiner='xor',
                    preprocessing='short',
                    layers=[layer_size, layer_size, layer_size],
                    activation='relu',
                    domain_in=-1,
                    learning_rate=learning_rate,
                    penalty=0.0002,
                    beta_1=0.9,
                    beta_2=0.999,
                    tolerance=0.0025,
                    patience=self.PATIENCE,
                    iteration_limit=self.ITERATION_LIMIT,
                    batch_size=batch_size,
                    print_learning=self.PRINT_LEARNING,
                )
            )
            for seed in range(10)
            for batch_size in [10**5]
            for layer_size in [128, 192, 256, 386, 512]
            for n in [64]
            for (k_up, k_down), (N_set, LR_set) in self.SIZES.items()
            for N in N_set
            for learning_rate in LR_set
        ]

    def plot(self):
        def get_size(row):
            if row.get('k_up', None) and not isnan(row['k_up']):
                return '(%i,%i)' % (int(row['k_up']), int(row['k_down']))
            elif row.get('transformation', None):
                try:
                    s = re.search('interpose k_down=([0-9]+) k_up=([0-9]+) transform=atf', row['transformation'])
                    return '(%i,%i)' % (int(s.group(1)), int(s.group(2)))
                except TypeError:
                    pass
            return float('nan')

        data = self.experimenter.results
        data['size'] = data.apply(lambda row: get_size(row), axis=1)
        data['max_memory_gb'] = data.apply(lambda row: row['max_memory'] / 1024**3, axis=1)
        data['Ne6'] = data.apply(lambda row: row['N'] / 1e6, axis=1)
        data = data.sort_values(['size', 'layers'])

        #self.old_plot(data)

        with axes_style('whitegrid'):
            params = dict(
                data=data,
                x='Ne6',
                y='accuracy',
                row='size',
                kind='swarm',
                aspect=7.5,
                height=1.2,
            )
            for name, params_ind in {
                'layer': dict(hue='layers', hue_order=[str([2**s] * 3) for s in range(2, 10)]),
                'learning_rate': dict(hue='learning_rate'),
            }.items():
                f = catplot(**params, **params_ind)
                f.savefig(f'figures/{self.name()}.{name}.pdf')
                close(f.fig)

    def old_plot(self, data):
        """
        Visualize the quality, process, and runtime of learning by plotting the accuracy, the accuracies of each epoch,
        and the measured time of each experiment, respectively.
        """
        self.plot_param_dependency(
            name='IPUF',
            df=data,
            param_x='learning_rate',
            param_1='N',
            param_2='batch_size',
        )
        """
        self.plot_history(
            df=self.experimenter.results
        )
        """
        self.plot_overview(df=data)

    def plot_param_dependency(self, name, df, param_x, param_1, param_2=None):
        param_y = 'accuracy'
        ncols = len(df.groupby(['n', 'size']).count())
        nrows = 3
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(9 * ncols, 4 * nrows)
        axes = axes.reshape((nrows, ncols))
        alpha_scatter = 0.3
        alpha_lines = 0.7
        palette = color_palette(palette='plasma')
        for j, (n, size) in enumerate(df.groupby(['n', 'size'])):
            print(j, n, size)
            data = df[(df['size'] == size) & (df['n'] == n)]
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
            )
            axes[0][j].set_title('n={}, size={}\n\n{} experiments\n'.format(
                n, size, len(data)))
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
            )
            axes[1][j].set_xscale('linear')
            axes[1][j].set_yscale('linear')
            axes[1][j].set_xlabel(param_y)
            axes[1][j].set_ylabel('runtime in s')
            axes[1][j].set_ylim(bottom=0)
            axes[1][j].xaxis.set_minor_locator(FixedLocator([.7, .9, .98]))
            axes[1][j].grid(b=True, which='minor', color='gray', linestyle=':')
            axes[1][j].legend(loc='upper right', bbox_to_anchor=(1.235, 1.03))

            lineplot(
                x='accuracy',
                y='max_memory',
                hue=param_1,
                style=param_2,
                data=data,
                ax=axes[2][j],
                legend='full',
                estimator='mean',
                ci=None,
                alpha=alpha_lines,
            )
            axes[2][j].set_xscale('linear')
            axes[2][j].set_yscale('linear')
            axes[2][j].set_xlabel(param_y)
            axes[2][j].set_ylabel('memory consumption')
            axes[2][j].set_ylim(bottom=0)
            axes[2][j].xaxis.set_minor_locator(FixedLocator([.7, .9, .98]))
            axes[2][j].grid(b=True, which='minor', color='gray', linestyle=':')
            axes[2][j].legend(loc='upper right', bbox_to_anchor=(1.235, 1.03))
        fig.subplots_adjust(hspace=.5, wspace=.5)
        title = fig.suptitle('Accuracy of {} Results on XOR Arbiter PUF'.format(name))
        title.set_position([.5, 1.05])
        fig.savefig('figures/{}_{}.pdf'.format(self.name(), name), bbox_inches='tight', dpi=300, pad_inches=.5)

        width = 0.3
        seed(42)
        df = self.experimenter.results
        distances = 1 - df.accuracy
        df['distance'] = distances
        ncols = len(set([k_up for n, k_up, k_down in self.SIZES.keys()]))
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
                             'using Multilayer Perceptron on each 100 PUF simulations per width k', size=16)
        title.set_position([0.5, 1.0])
        fig.savefig('figures/{}_overview.pdf'.format(self.name()), bbox_inches='tight', dpi=300, pad_inches=.5)

    def plot_history(self, df):
        ks = list(sorted(set(self.experimenter.results.k)))
        intervals = {(.0, .7): 'bad', (.7, .9): 'medium', (.9, .98): 'good', (.98, 1.): 'perfect'}
        sizes = [(k, n) for k in ks for n in list(sorted(set(df[df.k == k].n)))]
        ncols = len(sizes)
        nrows = len(intervals)
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(8 * ncols, 3 * nrows)
        axes = axes.reshape((nrows, ncols))
        for j, (k, n) in enumerate(sizes):
            data = df[(df.k == k) & (df.n == n)]
            axes[0][j].set_title('k={}, n={}\n'.format(k, n))
            for i, (low, high) in enumerate(intervals):
                curves = data[(data.accuracy >= low) & (data.accuracy < high)].accuracy_curve
                axes[i][j].set_ylabel('{} results\n\n{}'.format(intervals[(low, high)], 'accuracy') if j == 0
                                      else '{}'.format('accuracy'))
                axes[i][j].set_xlabel('{}'.format('epoch'))
                for curve in curves:
                    axes[i][j].plot([float(s) for s in re.findall(pattern=r'-?\d+\.?\d*', string=str(curve))])
        fig.subplots_adjust(hspace=.5, wspace=.5)
        title = fig.suptitle('History of {} of MLP learning Results on XOR Arbiter PUFs'.format('accuracy'))
        title.set_position([.5, 1.05])

        fig.savefig('figures/{}_history.pdf'.format(self.name()), bbox_inches='tight', pad_inches=.5)

    def plot_overview(self, df):
        seed(42)
        ncols = 1
        nrows = 2
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(7 * ncols, 4 * nrows)
        axes = axes.reshape((nrows, ncols))
        marker = 'o'
        for i, experiment in enumerate(sorted(list(set(df['experiment'])))):
            stripplot(
                x='k',
                y='accuracy',
                data=df[df['experiment'] == experiment],
                ax=axes[0][i],
                jitter=True,
                alpha=0.4,
                zorder=1,
                marker=marker,
            )
            means_accuracy = [df[(df.experiment == experiment) & (df.k == k)]['accuracy'].mean()
                              for k in sorted(list(set(df['k'])))]
            for j, k in enumerate(set(df.k)):
                axes[0][i].plot((-0.25 + j, 0.235 + j), 2 * (means_accuracy[j],),
                                linewidth=2, label=str(round(means_accuracy[j], 3)))
            axes[0][i].set_yscale('linear')
            axes[0][i].set_ylabel('accuracy')
            axes[0][i].legend(loc='upper right', bbox_to_anchor=(1.185, 1.02), title='means')

            stripplot(
                x='k',
                y='measured_time',
                data=df[df['experiment'] == experiment],
                ax=axes[1][i],
                jitter=True,
                alpha=0.4,
                zorder=1,
                marker=marker,
            )
            means_runtime = [df[(df.experiment == experiment) & (df.k == k)]['measured_time'].mean()
                             for k in sorted(list(set(df['k'])))]
            for j, k in enumerate(set(df.k)):
                axes[1][i].plot((-0.25 + j, 0.235 + j), 2 * (means_runtime[j],),
                                linewidth=2, label=str(int(round(means_runtime[j], 0))))
            axes[1][i].set_yscale('log')
            axes[1][i].set_ylabel('runtime in s')
            axes[1][i].legend(loc='upper right', bbox_to_anchor=(1.18, 1.02), title='means')
        fig.subplots_adjust(hspace=0.3, wspace=0.6)
        title = fig.suptitle('Overview of Learning Results\non 64-Bit (k,k)-IPUFs with MLP', size=20)
        title.set_position([0.5, 1.0])
        fig.savefig('figures/{}_ipuf_overview.pdf'.format(self.name()), bbox_inches='tight', pad_inches=.5)