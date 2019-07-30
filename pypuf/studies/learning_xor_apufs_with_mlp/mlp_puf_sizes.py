"""
This module describes a study that defines a set of experiments in order to find good hyperparameters and to examine
the quality of Deep Learning based modeling attacks on XOR Arbiter PUFs. Furthermore, some plots are defined to
visualize the experiment's results.
The Deep Learning technique used here is the Optimization technique Adam for a Stochastic Gradient Descent on a Feed-
Forward Neural Network architecture called Multilayer Perceptron (MLP). Implementations of the MLP and Adam are
used from Scikit-Learn and Tensorflow, respectively.
"""
from re import findall

from math import floor
from matplotlib.pyplot import subplots
from matplotlib.ticker import FixedLocator
from numpy.ma import ones, log10
from numpy.random.mtrand import seed
from seaborn import stripplot, lineplot, scatterplot

from pypuf.studies.base import Study
from pypuf.experiments.experiment.mlp_tf import ExperimentMLPTensorflow, Parameters as Parameters_tf
from pypuf.experiments.experiment.mlp_skl import ExperimentMLPScikitLearn, Parameters as Parameters_skl


class MLPPUFSizesStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """

    TRANSFORMATION = 'id'
    COMBINER = 'xor'
    PREPROCESSING = 'no'
    ACTIVATION = 'relu'
    ITERATION_LIMIT = 100
    LOSS = 'log_loss'
    DOMAIN_IN = -1
    PATIENCE = 4
    TOLERANCE = 0.0025
    PENALTY = 0.0002
    BETA_1 = 0.9
    BETA_2 = 0.999
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False

    PLOT_ESTIMATORS = []

    SIZES = {
        (32, 1): [4e3, 10e3, 25e3],   # [2e3, 4e3, 6e3, 8e3, 10e3, 12e3, 14e3, 16e3, 18e3, 20e3, 22e3, 24e3, 26e3, 28e3, 30e3],
        (32, 2): [4e3, 8e3, 12e3, 16e3, 20e3, 24e3, 28e3, 32e3, 36e3, 40e3, 44e3, 48e3, 52e3],
        (32, 3): [5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 35e3, 40e3, 45e3, 50e3, 55e3, 60e3, 65e3, 70e3, 75e3, 80e3],
        (64, 1): [2e3, 4e3, 6e3, 8e3, 10e3, 12e3, 14e3, 16e3, 18e3, 20e3, 22e3, 24e3, 26e3, 28e3, 30e3, 32e3, 34e3, 36e3],
        (64, 2): [5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 35e3, 40e3, 45e3, 50e3, 55e3, 60e3, 65e3, 70e3, 75e3, 80e3],
        (64, 3): [50e3, 150e3, 300e3],   # [25e3, 50e3, 75e3, 100e3, 125e3, 150e3, 175e3, 200e3, 225e3, 250e3, 275e3, 300e3],
        (128, 1): [3e3, 6e3, 9e3, 12e3, 15e3, 18e3, 21e3, 24e3, 27e3, 30e3, 33e3, 36e3, 39e3, 42e3, 45e3, 48e3, 51e3],
        (128, 2): [20e3, 40e3, 60e3, 80e3, 100e3, 120e3, 140e3, 160e3, 180e3, 200e3, 220e3, 240e3, 260e3],
        (128, 3): [80e3, 240e3, 520e3],    # [40e3, 80e3, 120e3, 160e3, 200e3, 240e3, 280e3, 320e3, 360e3, 400e3, 440e3, 480e3, 520e3],
        (256, 1): [4e3, 8e3, 12e3, 16e3, 20e3, 24e3, 28e3, 32e3, 36e3, 40e3, 44e3, 48e3, 52e3, 56e3, 60e3, 64e3],
        (256, 2): [10e3, 20e3, 40e3, 80e3, 120e3, 160e3, 200e3, 240e3, 280e3, 320e3, 360e3, 400e3, 440e3, 480e3, 520e3],
        (256, 3): [25e3, 50e3, 100e3, 150e3, 200e3, 250e3, 300e3, 350e3, 400e3, 450e3, 500e3, 550e3, 600e3, 650e3, 700e3],
        (512, 1): [12e3, 36e3, 90e3],   # [6e3, 12e3, 18e3, 24e3, 30e3, 36e3, 42e3, 48e3, 54e3, 60e3, 66e3, 72e3, 78e3, 84e3, 90e3],
        (512, 2): [25e3, 50e3, 100e3, 150e3, 200e3, 250e3, 300e3, 350e3, 400e3, 450e3, 500e3, 550e3, 600e3, 650e3, 700e3],
        (512, 3): [50e3, 100e3, 200e3, 400e3, 600e3, 800e3, 1000e3, 1200e3, 1400e3, 1600e3, 1800e3, 2000e3, 2200e3, 2400e3],
        (1024, 1): [8e3, 16e3, 24e3, 32e3, 40e3, 48e3, 56e3, 64e3, 72e3, 80e3, 88e3, 96e3, 104e3],
        (1024, 2): [25e3, 50e3, 100e3, 200e3, 400e3, 600e3, 800e3, 1000e3, 1200e3, 1400e3, 1600e3, 1800e3, 2000e3],
        (1024, 3): [50e3, 100e3, 200e3, 400e3, 800e3, 1200e3, 1600e3, 2000e3, 2400e3, 2800e3, 3200e3, 3600e3, 4000e3],
    }

    SAMPLES_PER_POINT = {
        (32, 1): 30,
        (32, 2): 30,
        (32, 3): 30,
        (64, 1): 30,
        (64, 2): 30,
        (64, 3): 30,
        (128, 1): 30,
        (128, 2): 30,
        (128, 3): 30,
        (256, 1): 30,
        (256, 2): 30,
        (256, 3): 30,
        (512, 1): 30,
        (512, 2): 30,
        (512, 3): 30,
        (1024, 1): 30,
        (1024, 2): 30,
        (1024, 3): 30,
    }

    LAYERS = {
        (32, 1): [[2**1, 2**1, 2**1]],
        (32, 2): [[2**2, 2**2, 2**2]],
        (32, 3): [[2**3, 2**3, 2**3]],
        (64, 1): [[2**1, 2**1, 2**1]],
        (64, 2): [[2**2, 2**2, 2**2]],
        (64, 3): [[2**3, 2**3, 2**3]],
        (128, 1): [[2**1, 2**1, 2**1]],
        (128, 2): [[2**2, 2**2, 2**2]],
        (128, 3): [[2**3, 2**3, 2**3]],
        (256, 1): [[2**1, 2**1, 2**1]],
        (256, 2): [[2**2, 2**2, 2**2]],
        (256, 3): [[2**3, 2**3, 2**3]],
        (512, 1): [[2**1, 2**1, 2**1]],
        (512, 2): [[2**2, 2**2, 2**2]],
        (512, 3): [[2**3, 2**3, 2**3]],
        (1024, 1): [[2**1, 2**1, 2**1]],
        (1024, 2): [[2**2, 2**2, 2**2]],
        (1024, 3): [[2**3, 2**3, 2**3]],
    }

    LEARNING_RATES = {
        (32, 1): [0.01, 0.014, 0.018, 0.022, 0.026, 0.03, 0.034, 0.037, 0.038, 0.039, 0.04],
        (32, 2): [0.0475],
        (32, 3): [0.0182],
        (64, 1): [0.088],
        (64, 2): [0.0255],
        (64, 3): [0.01, 0.012, 0.014, 0.016, 0.018, 0.019, 0.02, 0.027, 0.028, 0.029, 0.03, 0.032, 0.034, 0.036, 0.04, 0.045, 0.05, 0.06, 0.07],
        (128, 1): [0.083],
        (128, 2): [0.024],
        (128, 3): [0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01, 0.0105, 0.011, 0.0115, 0.012],
        (256, 1): [0.06],
        (256, 2): [0.0125],
        (256, 3): [0.0115],
        (512, 1): [0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.0515, 0.052, 0.0525, 0.053, 0.077, 0.0775, 0.078, 0.0785, 0.079],
        (512, 2): [0.005],
        (512, 3): [0.0048],
        (1024, 1): [0.031],
        (1024, 2): [0.005],
        (1024, 3): [0.003],
    }

    EXPERIMENTS = []

    def experiments(self):
        """
        Generate an experiment for every parameter combination corresponding to the definitions above.
        For each combination of (size, samples_per_point, learning_rate) different random seeds are used.
        """
        for c1, (n, k) in enumerate(self.SIZES.keys()):
            for N in self.SIZES[(n, k)]:
                for c2 in range(self.SAMPLES_PER_POINT[(n, k)]):
                    for c3, learning_rate in enumerate(self.LEARNING_RATES[(n, k)]):
                        cycle = c1 * (self.SAMPLES_PER_POINT[(n, k)] * len(self.LEARNING_RATES[(n, k)])) \
                                + c2 * len(self.LEARNING_RATES[(n, k)]) + c3
                        validation_frac = max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N
                        for layers in self.LAYERS[(n, k)]:
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
                                        domain_in=self.DOMAIN_IN,
                                        learning_rate=learning_rate,
                                        penalty=self.PENALTY,
                                        beta_1=self.BETA_1,
                                        beta_2=self.BETA_2,
                                        tolerance=self.TOLERANCE,
                                        patience=self.PATIENCE,
                                        iteration_limit=self.ITERATION_LIMIT,
                                        batch_size=1000 if k < 6 else 10000,
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
        pass
        if not self.EXPERIMENTS:
            self.experiments()
        """
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
        """
        param_y = 'accuracy'
        df['layers'] = df['layers'].apply(str)
        df = df[df['experiment'] == 'ExperimentMLP' + name]
        ncols = len(self.SIZES.keys())
        nrows = 2
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(9 * ncols, 4 * nrows)
        axes = axes.reshape((nrows, ncols))
        for j, (n, k) in enumerate(self.SIZES.keys()):
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
            )
            scatterplot(
                x=param_x,
                y=param_y,
                hue=param_1,
                style=param_2,
                data=data,
                ax=axes[0][j],
                legend='full',
            )
            lib = 'tensorflow' if name == 'Tensorflow' else 'scikit-learn' if name == 'ScikitLearn' else ''
            total = sum([e.parameters.k == k and e.parameters.n == n and lib in e.NAME for e in self.EXPERIMENTS])
            axes[0][j].set_title('n={}, k={}\n\n{} experiments per combination,   {}/{}\n'.format(
                n, k, self.SAMPLES_PER_POINT[(n, k)], len(data), total))
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
        fig.savefig('figures/{}_{}.pdf'.format(self.name(), name), bbox_inches='tight', pad_inches=.5)
        """

        def pypuf_round(x, p):
            precision = -floor(log10(x)) + p
            return round(x, precision) if precision > 0 else int(round(x, 0))
        alpha = 0.1
        width = 0.3
        seed(42)
        df = self.experimenter.results
        distances = 1 - df.accuracy
        df['distance'] = distances
        ncols = len(set([k for n, k in self.SIZES.keys()]))
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
                alpha=alpha,
                zorder=1,
                marker=marker,
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
            #axes[0][i].legend(loc='upper right', bbox_to_anchor=(1.2, 1.02), title='means')

            stripplot(
                x='n',
                y='distance',
                data=df[df.k == k],
                ax=axes[1][i],
                jitter=True,
                alpha=alpha,
                zorder=1,
                marker=marker,
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
            #axes[1][i].legend(loc='upper right', bbox_to_anchor=(1.2, 1.02), title='means')

            stripplot(
                x='n',
                y='measured_time',
                data=df[df.k == k],
                ax=axes[2][i],
                jitter=True,
                alpha=alpha,
                zorder=1,
                marker=marker,
            )
            for j, n in enumerate(list(sorted(set(df.n)))):
                Ns = list(sorted(set(df[(df.k == k) & (df.n == n)].N)))
                for l, N in enumerate(Ns):
                    mean = df[(df.k == k) & (df.n == n) & (df.N == N)].measured_time.mean()
                    axes[2][i].plot([j - width / 2, j + width / 2], [mean, mean],
                                    color=(l / len(Ns),) * 3, linewidth=1, zorder=2, label=l if j == 0 else None)
            axes[2][i].set_yscale('log')
            #axes[2][i].legend(loc='upper right', bbox_to_anchor=(1.2, 1.02), title='means')

        axes[2][0].set_ylabel('runtime in s')
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        title = fig.suptitle('Overview of Learning Results on XOR Arbiter PUFs of length 64\n'
                             'using Multilayer Perceptron on each 100 PUF simulations per width k', size=16)
        title.set_position([0.5, 1.0])
        fig.savefig('figures/{}_overview.pdf'.format(self.name()), bbox_inches='tight', pad_inches=.5)

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
            #total = sum([e.parameters.k == k and lib in e.NAME for e in self.EXPERIMENTS])
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

