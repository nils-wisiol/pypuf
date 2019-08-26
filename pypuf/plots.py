"""
Plots to visualize results by experiments or studies.
"""

from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from numpy import max as max_np, min as min_np, log, array, log2, log10
from numpy import mean as mean_np, quantile as quantile_np, sum as sum_np
from numpy import zeros
from pandas import read_csv, DataFrame
from seaborn import lineplot, set_style


class SuccessRatePlot:
    """
    Show the ratio of experiment results with accuracy higher than a given threshold to the total number of results,
    dependent on the number of examples in the training set.
    """

    def __init__(self, filename, group_by, experiment_hashes=None, success_threshold=.7, group_labels=None):
        """
        Prepare a plot
        :param filename: destination file (PDF)
        :param group_by: determines among which groups success rates are computed
        :param experiment_hashes: ids of results that shall be used
        :param success_threshold: defines what is considered a success
        :param group_labels: can be used to translate group_by values to human-readable names, e.g.
            { 'permutation_fixed': 'A fixed permutation', ... }
        """
        self.title_size = 6
        self.tick_size = 4
        self.x_label_size = 5
        self.legend_size = 4

        self.n = None
        self.k = None
        self.experiment = None
        self.title = None
        self.results = None

        self.filename = filename
        self.experiment_hashes = experiment_hashes or []
        self.success_threshold = success_threshold
        self.group_by = group_by
        self.group_labels = {} if group_labels is None else group_labels

        self.figure = plt.figure()
        self.figure.set_size_inches(w=3.34, h=0.8)
        self.axis = self.figure.add_subplot(1, 1, 1)

        self.plot_data = None

    def plot(self, results):
        """
        Draw the plot and save it to the file system.
        """
        if self.experiment_hashes:
            results = results.loc[results['experiment_hash'].isin(self.experiment_hashes)]

        if results.empty:
            return

        self.axis.clear()

        self.axis.set_xlabel('number of examples in the training set', size=self.x_label_size)
        # self.axis.set_ylabel('success rate (threshold: %f)' % success_threshold, size=ylabelsize)
        for w in ['major', 'minor']:
            self.axis.tick_params(width=0.5, which=w, labelsize=self.tick_size)
        for axis in ['top', 'bottom', 'left', 'right']:
            self.axis.spines[axis].set_linewidth(0.5)

        n_k_combinations = results.groupby(['n', 'k']).size().reset_index().values[:, :2]
        if not n_k_combinations.size:
            return
        assert len(n_k_combinations) == 1, \
            "For SuccesRatePlot, all experiments must be run with same n and k, but there were %s." % n_k_combinations
        self.n = results['n'].unique()[0]
        self.k = results['k'].unique()[0]
        self.title = 'Success Rate on %i-bit, %i-XOR Arbiter PUF' % (
            self.n,
            self.k,
        )
        self.axis.set_title(self.title, size=self.title_size)

        color = cycle([plt.cm.colors.rgb2hex(x) for x in [
            "#F44336",  # Red
            "#4CAF50",  # Green
            "#03A9F4",  # Indigo
            "#FFC107",  # Amber
            "#795548",  # Brown
        ]])

        marker = cycle(["8", "s", "^", "x", "D"])

        self.plot_data = {}

        self.axis.set_xscale("log")
        min_N = results['N'].min()
        max_N = results['N'].max()
        if min_N == max_N:
            max_N += 1
        self.axis.set_xlim([min_N, max_N])
        self.axis.set_ylim([-.02, 1.02])

        for name, group_results in results.groupby([self.group_by]):
            Ns = group_results['N'].unique()
            success_rate = zeros((len(Ns), 2))
            for idx, N in enumerate(Ns):
                group_N_results = group_results[group_results['N'] == N]
                success_rate[idx, 0] = N
                success_rate[idx, 1] = \
                    group_N_results[
                        group_N_results['accuracy'] > self.success_threshold
                        ].size / group_N_results.size
            success_rate.sort(axis=0)
            self.plot_data[name] = success_rate

            label = self.group_labels[name] if name in self.group_labels else name
            col = next(color)
            mar = next(marker)
            self.axis.scatter(success_rate[:, 0], success_rate[:, 1], 8, color=col, label=label, marker=mar, alpha=.7)
            self.axis.plot(success_rate[:, 0], success_rate[:, 1], '-', color=col, linewidth=0.8, alpha=.7)

        if self.axis.has_data():
            legend = self.axis.legend(loc='best', fontsize=self.legend_size)
            self.figure.savefig(self.filename, bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0)


class AccuracyPlotter:

    def __init__(self, min_tick, max_tick, estimator='mean', group_by='transformation', group_by_ex=None, grid=False):
        self.estimator = estimator
        self.group_by = group_by
        self.group_by_ex = group_by_ex
        self.min_tick = min_tick
        self.max_tick = max_tick
        self.grid = grid
        self.df = None

    def get_data_frame(self, source, names=None):
        self.df = read_csv(source, names=names, sep='\t', header=None) if isinstance(source, str) else source
        return

    def get_title(self):
        k = self.df['k'].iloc[0]
        n = self.df['n'].iloc[0]
        return 'Comparison of Accuracies \non ({}, {})-XOR-Arbiter PUFs \n'.format(k, n) + \
               'using {} as estimator'.format('{} with p={}'.format(self.estimator[0], self.estimator[1])
                                              if isinstance(self.estimator, tuple) else 'mean')

    def create_plot(self, save_path=None):
        figure = plt.figure()
        axis = figure.add_subplot()
        assert isinstance(self.df, DataFrame)
        set_style('white')
        estimator = get_estimator(self.estimator)
        figure = lineplot(
            x='N',
            y='accuracy',
            hue=self.group_by,
            style=self.group_by_ex,
            estimator=estimator,
            ci=None,
            data=self.df,
            alpha=0.6,
            sort=True,
        )
        legend_alpha = 1
        axis.set_xlabel(
            '{} with p={}'.format(self.estimator[0], self.estimator[1])
            if isinstance(self.estimator, tuple) else 'mean'
        )
        axis.set_ylim([-.01 if isinstance(self.estimator, tuple) and self.estimator[0] == 'success' else .49, 1.01])
        axis.set_xlim([0, self.max_tick + (self.max_tick / 100)])
        axis.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        axis.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
        axis.tick_params(which='major', width=1.0, length=5)
        axis.tick_params(which='minor', width=0.5, length=2)
        if self.grid:
            plt.grid(True)
            legend_alpha = 0.5
            axis.xaxis.set_minor_locator(plt.MultipleLocator(self.min_tick))
            axis.grid(b=True, which='major', color='lightgrey', linewidth=1)
            axis.grid(b=True, which='minor', color='lightgrey', linewidth=0.5)
        handles, labels = axis.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        legend = axis.legend(handles, labels, loc='best', framealpha=legend_alpha)
        for l in legend.get_lines():
            l.set_alpha(0.7)
        axis.set_title(self.get_title())
        if save_path is not None:
            figure.savefig(fname=save_path, dpi=500, quality=95, format='pdf')
        plt.close()
        return


def get_estimator(estimator):
    if estimator == 'mean':
        return mean_np

    elif estimator == 'max' or estimator == 'best':
        return max_np

    elif estimator == 'min' or estimator == 'worst':
        return min_np

    elif estimator.startswith('quantile'):
        q = float(estimator[estimator.find('=')+1:]) if '=' in estimator else 0.9

        def quantile(values):
            return quantile_np(values, q, interpolation='linear')

        return quantile

    elif estimator.startswith('success'):
        p = float(estimator[estimator.find('=')+1:]) if '=' in estimator else 0.7

        def success(values):
            return sum_np(values >= p) / len(values)

        return success

    return


class PermutationIndexPlot:
    """
    Show the distribution of indices of the permutations which lead to a successful learning of the instance using
    the correlation attack.
    """

    def __init__(self, filename, experiment_hashes=None, group_labels=None, group_subplot_layout=None,
                 w=3.34, h=1.7):
        """

        :param filename: destination file (PDF)
        :param group_by: Sets the group_by column. Each unique value of this column creates an individual subplot.
        :param experiment_hashes: hashes of experiments that shall be used
        :param group_labels: Titles of subplots as dict (key=group_by column value, value=label)
        :param group_subplot_layout: Layout of subplots as dict (key=group_by column value, value=subplot layout tuple)
        :param w: Width of the resulting figure (in inches)
        :param h: Height of the resulting figure (in inches)
        """
        self.title_size = 4
        self.tick_size = 3
        self.x_label_size = 4
        self.legend_size = 3

        self.n = None
        self.k = None
        self.results = None

        self.filename = filename
        self.experiment_hashes = experiment_hashes
        self.group_labels = {} if group_labels is None else group_labels

        self.figure = plt.figure()
        self.figure.set_size_inches(w=w, h=h)
        self.group_subplot_layout = group_subplot_layout

        self.plot_data = None

    def plot(self, results):
        """
        Draw the plot and save it to the file system.
        """
        if self.experiment_hashes:
            results = results.loc[results['experiment_hash'].isin(self.experiment_hashes)]

        if results.empty:
            return

        self.figure.clear()

        axis, i = None, 0
        for params in self.group_subplot_layout:
            group_results = results[
                (results['n'] == params.n) &
                (results['k'] == params.k) &
                (results['N'] == params.N)
            ]

            if not params.plot_layout:
                continue
            axis = self.figure.add_subplot(*params.plot_layout)
            axis.tick_params(width=0.25, which='both', labelsize=self.tick_size, direction='in')
            for a in ['top', 'bottom', 'left', 'right']:
                axis.spines[a].set_linewidth(0.5)
            axis.set_title(params.label, size=self.title_size)
            successful_runs = group_results[group_results['best_permutation_iteration'] > 0]
            permutation_indices = successful_runs[['best_permutation_iteration',
                                                   'total_permutation_iterations']].max(axis=1).tolist()
            if not permutation_indices:
                continue
            permutation_indices.sort()
            m = int(max(permutation_indices))
            amounts, _, _ = axis.hist(permutation_indices, density=True, label='Rel. Frequency', bins=range(1, m + 1),
                                      histtype='step', linestyle='--', linewidth=0.5)

            top_pos = None
            xs = [1]
            ys = [0]
            s = 0
            for idx, amount in enumerate(amounts):
                s += amount
                xs.append(idx + 2)
                ys.append(s)
                if not top_pos and s >= 0.8:
                    top_pos = idx + 2

            if top_pos:
                axis.plot((top_pos, top_pos), (0, 0.8), linewidth=0.25)
            axis.axhline(y=0.8, linewidth=0.25)
            axis.plot(xs, ys, label='Cum. Rel. Frequency', linewidth=0.5)
            axis.scatter(top_pos, 0.8, s=2, zorder=5)

            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.xaxis.set_minor_locator(MultipleLocator(base=1))
            axis.yaxis.set_minor_locator(MultipleLocator(base=0.05))

            if i == len(self.group_subplot_layout) // 2:
                axis.legend(loc=7, fontsize=self.legend_size)
            i += 1

        plt.tight_layout()
        self.figure.savefig(self.filename)


def plot3d_size_dependency(df):
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    xs = log2(df.n)
    ys = df.k
    zs1 = log10(df.x_N)
    ax1.scatter(xs, ys, zs1)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([rf'$2^{ {int(x)} }$' for x in xs])
    ax1.set_yticks(ys)
    # ax1.set_zticklabels([rf'${int(round(2**(z-10), 0))} \times 2^{ {10} }$' for z in ax1.get_zticks()[1:]])
    ax1.set_zticklabels([rf'${int(round(10**(z-3), 0))} \times 10^3$' for z in ax1.get_zticks()[1:]])
    ax1.set_xlabel('n')
    ax1.set_ylabel('k')
    ax1.set_zlabel('N')
    ax1.view_init(azim=240, elev=20)
    numx = len(set(xs))
    numy = len(set(ys))
    for i in range(numy):
        indices = list(i * numx + array(list(range(numx))))
        ax1.plot(xs[indices], ys[indices], zs1[indices], color='b', alpha=0.5)
    for i in range(numx):
        indices = list(numx * array(list(range(numy))) + i)
        ax1.plot(xs[indices], ys[indices], zs1[indices], color='b', alpha=0.5)

    ax2 = fig.add_subplot(122, projection='3d')
    zs2 = df.x_learning_rate
    ax2.scatter(xs, ys, zs2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([rf'$2^{ {int(x)} }$' for x in xs])
    ax2.set_yticks(ys)
    ax2.set_xlabel('n')
    ax2.set_ylabel('k')
    ax2.set_zlabel('learning_rate')
    ax2.view_init(azim=330, elev=20)
    for i in range(numy):
        indices = list(i * numx + array(list(range(numx))))
        ax2.plot(xs[indices], ys[indices], zs2[indices], color='b', alpha=0.5)
    for i in range(numx):
        indices = list(numx * array(list(range(numy))) + i)
        ax2.plot(xs[indices], ys[indices], zs2[indices], color='b', alpha=0.5)
    fig.subplots_adjust(hspace=0)
    fig.savefig('figures/sizes_dependency.pdf')
