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

from pypuf.experiments.experiment.mlp_tf_ipuf import ExperimentInterposeMLPTensorflow, Parameters
from pypuf.studies.base import Study
from pypuf.experiments.experiment.mlp_skl import ExperimentMLPScikitLearn, Parameters as Parameters_skl


class InterposeMLPStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """

    ITERATION_LIMIT = 80
    PATIENCE = 8
    TOLERANCE = 0.0025
    PENALTY = 0.0002
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False

    EXPERIMENTS = []

    SCALES = [1.0]

    SIZES = {
        (64, 2, 2): list(map(lambda x: x*200e3, SCALES)),
        (64, 3, 3): list(map(lambda x: x*240e3, SCALES)),
        (64, 4, 4): list(map(lambda x: x*320e3, SCALES)),
        (64, 5, 5): list(map(lambda x: x*1.2e6, SCALES)),
        (64, 6, 6): list(map(lambda x: x*8e6, SCALES)),
    }

    SAMPLES_PER_POINT = {
        (64, 2, 2): 10,
        (64, 3, 3): 10,
        (64, 4, 4): 10,
        (64, 5, 5): 10,
        (64, 6, 6): 3,
    }

    LEARNING_RATES = {
        (64, 2, 2): [0.0002],
        (64, 3, 3): [0.0002],
        (64, 4, 4): [0.0002],
        (64, 5, 5): [0.0002],
        (64, 6, 6): [0.0002],
    }

    BATCH_SIZES = {
        (64, 2, 2): [2**14],
        (64, 3, 3): [2**14],
        (64, 4, 4): [2**14],
        (64, 5, 5): [2**14],
        (64, 6, 6): [2**14],
    }

    TRANSFORMATIONS = {
        (64, 2, 2): ['atf', 'lightweight_secure', 'fixed_permutation'],
        (64, 3, 3): ['atf', 'lightweight_secure', 'fixed_permutation'],
        (64, 4, 4): ['atf', 'lightweight_secure', 'fixed_permutation'],
        (64, 5, 5): ['atf', 'lightweight_secure', 'fixed_permutation'],
        (64, 6, 6): ['atf', 'lightweight_secure', 'fixed_permutation'],
    }

    PREPROCESSINGS = {
        (64, 2, 2): ['no', 'short', 'full'],
        (64, 3, 3): ['no', 'short', 'full'],
        (64, 4, 4): ['no', 'short', 'full'],
        (64, 5, 5): ['no', 'short', 'full'],
        (64, 6, 6): ['no', 'short', 'full'],
    }

    LAYERS = {
         (64, 2, 2): [[20 + 10*2]*3],
         (64, 3, 3): [[20 + 10*3]*3],
         (64, 4, 4): [[20 + 10*4]*3],
         (64, 5, 5): [[20 + 10*5]*3],
         (64, 6, 6): [[20 + 10*6]*3],
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
                            for transformation in self.TRANSFORMATIONS[(n, k_up, k_down)]:
                                for preprocessing in self.PREPROCESSINGS[(n, k_up, k_down)]:
                                    for layers in self.LAYERS[(n, k_up, k_down)]:
                                        """
                                        self.EXPERIMENTS.append(
                                            ExperimentInterposeMLPTensorflow(
                                                progress_log_prefix=None,
                                                parameters=Parameters(
                                                    seed_simulation=0x3 + cycle,
                                                    seed_challenges=0x1415 + cycle,
                                                    seed_model=0x9265 + cycle,
                                                    seed_distance=0x3589 + cycle,
                                                    n=n,
                                                    k_up=k_up,
                                                    k_down=k_down,
                                                    pos=n//2,
                                                    N=int(N),
                                                    validation_frac=validation_frac,
                                                    learning_rate=learning_rate,
                                                    penalty=self.PENALTY,
                                                    tolerance=self.TOLERANCE,
                                                    patience=self.PATIENCE,
                                                    iteration_limit=self.ITERATION_LIMIT,
                                                    batch_size=batch_size,
                                                    print_learning=self.PRINT_LEARNING,
                                                )
                                            )
                                        )
                                        """
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
        """
        if not self.EXPERIMENTS:
            self.experiments()
        self.plot_helper(
            name='ScikitLearn',
            df=self.experimenter.results,
            param_x='learning_rate',
            param_1='N',
            param_2=None,
        )
        self.plot_history(
            df=self.experimenter.results
        )
        """

    def plot_helper(self, name, df, param_x, param_1, param_2=None):
        param_y = 'accuracy'
        #df['layers'] = df['layers'].apply(str)
        #df = df[df['experiment'] == 'ExperimentMLP' + name]
        ncols = len(self.SIZES.keys())
        nrows = 2
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(9 * ncols, 4 * nrows)
        axes = axes.reshape((nrows, ncols))
        alpha_scatter = 0.1
        alpha_lines = 0.7
        palette = color_palette(palette='plasma')
        for j, (n, k_up, k_down) in enumerate(self.SIZES.keys()):
            data = df[(df['k_up'] == k_up) & (df['k_down'] == k_down) & (df['n'] == n)]
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
            lib = 'tensorflow' if name == 'Tensorflow' else 'scikit-learn' if name == 'ScikitLearn' else ''
            axes[0][j].set_title('n={}, k_up={}\n\n{} experiments\n'.format(
                n, k_up, len(data)))
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
        ncols = len(set([k_up for n, k_up, k_down in self.SIZES.keys()]))
        nrows = 3
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(7 * ncols, 4 * nrows)
        axes = axes.reshape((nrows, ncols))
        marker = 'o'

        for i, k_up in enumerate(list(sorted(set(df.k_up)))):
            stripplot(
                x='n',
                y='accuracy',
                data=df[df.k_up == k_up],
                ax=axes[0][i],
                jitter=True,
                zorder=1,
                marker=marker,
                alpha=alpha_scatter,
                #palette=palette,
            )
            for j, n in enumerate(list(sorted(set(df.n)))):
                Ns = list(sorted(set(df[(df.k_up == k_up) & (df.n == n)].N)))
                for l, N in enumerate(Ns):
                    mean = df[(df.k_up == k_up) & (df.n == n) & (df.N == N)].accuracy.mean()
                    axes[0][i].plot([j - width / 2, j + width / 2], [mean, mean],
                                    color=(l / len(Ns),) * 3, linewidth=1, zorder=2, label=l if j == 0 else None)
            axes[0][i].set_title('k={}\n'.format(k_up))
            axes[0][i].set_yscale('linear')
            axes[0][i].set_ylabel('accuracy')

            stripplot(
                x='n',
                y='distance',
                data=df[df.k_up == k_up],
                ax=axes[1][i],
                jitter=True,
                zorder=1,
                marker=marker,
                alpha=alpha_scatter,
                #palette=palette,
            )
            for j, n in enumerate(list(sorted(set(df.n)))):
                Ns = list(sorted(set(df[(df.k_up == k_up) & (df.n == n)].N)))
                for l, N in enumerate(Ns):
                    mean = df[(df.k_up == k_up) & (df.n == n) & (df.N == N)].distance.mean()
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
                data=df[df.k_up == k_up],
                ax=axes[2][i],
                jitter=True,
                zorder=1,
                marker=marker,
                alpha=alpha_scatter,
                #palette=palette,
            )
            for j, n in enumerate(list(sorted(set(df.n)))):
                Ns = list(sorted(set(df[(df.k_up == k_up) & (df.n == n)].N)))
                for l, N in enumerate(Ns):
                    mean = df[(df.k_up == k_up) & (df.n == n) & (df.N == N)].measured_time.mean()
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

