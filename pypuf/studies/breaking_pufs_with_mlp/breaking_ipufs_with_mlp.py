"""
This module describes a study that defines a set of experiments in order to find good hyperparameters and to examine
the quality of Deep Learning based modeling attacks on XOR Arbiter PUFs. Furthermore, some plots are defined to
visualize the experiment's results.
The Deep Learning technique used here is the Optimization technique Adam for a Stochastic Gradient Descent on a Feed-
Forward Neural Network architecture called Multilayer Perceptron (MLP). Implementations of the MLP and Adam are
used from Scikit-Learn and Tensorflow, respectively.
"""
from re import findall
from matplotlib.pyplot import subplots
from matplotlib.ticker import FixedLocator
from numpy.ma import ones
from numpy.random.mtrand import seed
from seaborn import stripplot, lineplot, scatterplot, color_palette

from pypuf.studies.base import Study
from pypuf.experiments.experiment.mlp_skl import ExperimentMLPScikitLearn, Parameters as Parameters_skl


class InterposeMLPStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """
    SHUFFLE = True

    ITERATION_LIMIT = 300
    PATIENCE = ITERATION_LIMIT
    TOLERANCE = 0.0025
    PENALTY = 0.0002
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False

    EXPERIMENTS = []

    SCALES = [0.5, 1.0, 2.0]

    SIZES = {
        # (64, 2, 2): list(map(lambda x: x*920e3, SCALES)),
        # (64, 3, 3): list(map(lambda x: x*1.2e6, SCALES)),
        # (64, 4, 4): list(map(lambda x: x*2e6, SCALES)),
        # (64, 5, 5): list(map(lambda x: x*4.8e6, SCALES)),
        # (64, 6, 6): list(map(lambda x: x*16e6, SCALES)),
        # (64, 7, 7): list(map(lambda x: x*80e6, SCALES)),
        (64, 1, 5): list(map(lambda x: x*600e3, SCALES)),
        (64, 1, 6): list(map(lambda x: x*2e6, SCALES)),
        (64, 1, 7): list(map(lambda x: x*10e6, SCALES)),
        (64, 1, 8): list(map(lambda x: x*10e6, SCALES)),
    }

    SAMPLES_PER_POINT = {
        (64, 2, 2): 10,
        (64, 3, 3): 10,
        (64, 4, 4): 10,
        (64, 5, 5): 10,
        (64, 6, 6): 5,
        (64, 7, 7): 5,
        (64, 1, 5): 10,
        (64, 1, 6): 10,
        (64, 1, 7): 10,
        (64, 1, 8): 10,
    }

    LEARNING_RATES = {
        (64, 2, 2): [0.002, 0.003, 0.004, 0.006, 0.008],
        (64, 3, 3): [0.002, 0.003, 0.004, 0.006, 0.008],
        (64, 4, 4): [0.0012, 0.0016, 0.002, 0.0024, 0.0028],
        (64, 5, 5): [0.0003, 0.0006, 0.0009, 0.0012, 0.0015],
        (64, 6, 6): [0.0002, 0.0004, 0.0006, 0.0008, 0.001],
        (64, 7, 7): [0.0002, 0.0004, 0.0006, 0.0008, 0.001],
        (64, 1, 5): [0.0005, 0.002, 0.008],
        (64, 1, 6): [0.0003, 0.0015, 0.006],
        (64, 1, 7): [0.0002, 0.001, 0.005],
        (64, 1, 8): [0.0001, 0.0008, 0.004],
    }

    BATCH_SIZES = {
        (64, 2, 2): [10**3, 10**4],
        (64, 3, 3): [10**3, 10**4],
        (64, 4, 4): [10**3, 10**4, 10**5],
        (64, 5, 5): [10**3, 10**4, 10**5],
        (64, 6, 6): [10**4, 10**5],
        (64, 7, 7): [10**4, 10**5],
        (64, 1, 5): [10**3, 10**4],
        (64, 1, 6): [10**3, 10**4],
        (64, 1, 7): [10**3, 10**4, 10**5],
        (64, 1, 8): [10**3, 10**4, 10**5],
    }

    TRANSFORMATIONS = ['atf']

    PREPROCESSINGS = ['short']

    LAYERS = {
        (64, 2, 2): [[2**2]*3, [2**3]*3, [2**4]*3],
        (64, 3, 3): [[2**3]*3, [2**4]*3, [2**5]*3],
        (64, 4, 4): [[2**4]*3, [2**5]*3, [2**6]*3],
        (64, 5, 5): [[2**5]*3, [2**6]*3, [2**7]*3],
        (64, 6, 6): [[2**6]*3, [2**7]*3, [2**8]*3],
        (64, 7, 7): [[2**7]*3, [2**8]*3, [2**9]*3],
        (64, 1, 5): [[2**5]*3, [2**6]*3, [2**7]*3],
        (64, 1, 6): [[2**6]*3, [2**7]*3, [2**8]*3],
        (64, 1, 7): [[2**7]*3, [2**8]*3, [2**9]*3],
        (64, 1, 8): [[2**8]*3, [2**9]*3, [2**10]*3],
    }

    def experiments(self):
        """
        Generate an experiment for every parameter combination corresponding to the definitions above.
        For each combination of (size, samples_per_point, learning_rate) different random seeds are used.
        """
        for c1, (n, k_up, k_down) in enumerate(self.SIZES.keys()):
            for N in self.SIZES[(n, k_up, k_down)]:
                validation_frac = max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N
                for c2 in range(self.SAMPLES_PER_POINT[(n, k_up, k_down)]):
                    for c3, learning_rate in enumerate(self.LEARNING_RATES[(n, k_up, k_down)]):
                        cycle = c1 * (self.SAMPLES_PER_POINT[(n, k_up, k_down)]
                                      * len(self.LEARNING_RATES[(n, k_up, k_down)])) \
                                + c2 * len(self.LEARNING_RATES[(n, k_up, k_down)]) + c3
                        for batch_size in self.BATCH_SIZES[(n, k_up, k_down)]:
                            for transformation in self.TRANSFORMATIONS:
                                for preprocessing in self.PREPROCESSINGS:
                                    for layers in self.LAYERS[(n, k_up, k_down)]:
                                        self.EXPERIMENTS.append(
                                            ExperimentMLPScikitLearn(
                                                progress_log_prefix=None,
                                                parameters=Parameters_skl(
                                                    seed_simulation=0x3 + cycle,
                                                    seed_challenges=0x1415 + cycle,
                                                    seed_model=0x9265 + cycle,
                                                    seed_distance=0x3589 + cycle,
                                                    n=n,
                                                    k=k_down,
                                                    N=int(N),
                                                    validation_frac=validation_frac,
                                                    transformation='interpose k_down={} k_up={} transform={}'.format(
                                                        k_down, k_up, transformation),
                                                    combiner='xor',
                                                    preprocessing=preprocessing,
                                                    layers=layers,
                                                    activation='relu',
                                                    domain_in=-1,
                                                    learning_rate=learning_rate,
                                                    penalty=self.PENALTY,
                                                    beta_1=0.9,
                                                    beta_2=0.999,
                                                    tolerance=self.TOLERANCE,
                                                    patience=self.PATIENCE,
                                                    iteration_limit=self.ITERATION_LIMIT,
                                                    batch_size=batch_size,
                                                    print_learning=self.PRINT_LEARNING,
                                                )
                                            )
                                        )
        return self.EXPERIMENTS

    def plot(self):
        """
        Visualize the quality, process, and runtime of learning by plotting the accuracy, the accuracies of each epoch,
        and the measured time of each experiment, respectively.
        """
        if not self.EXPERIMENTS:
            self.experiments()
        #"""
        self.plot_param_dependency(
            name='IPUF',
            df=self.experimenter.results,
            param_x='learning_rate',
            param_1='N',
            param_2='batch_size',
        )
        """
        self.plot_history(
            df=self.experimenter.results
        )
        """
        self.plot_overview(df=self.experimenter.results)

    def plot_param_dependency(self, name, df, param_x, param_1, param_2=None):
        param_y = 'accuracy'
        ncols = len(self.SIZES.keys())
        nrows = 3
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(9 * ncols, 4 * nrows)
        axes = axes.reshape((nrows, ncols))
        alpha_scatter = 0.3
        alpha_lines = 0.7
        palette = color_palette(palette='plasma')
        for j, (n, k_up, k_down) in enumerate(self.SIZES.keys()):
            data = df[(df['k'] == k_down) & (df['n'] == n)]
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
            axes[0][j].set_title('n={}, k_down={}\n\n{} experiments\n'.format(
                n, k_down, len(data)))
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
                    axes[i][j].plot([float(s) for s in findall(pattern=r'-?\d+\.?\d*', string=str(curve))])
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
