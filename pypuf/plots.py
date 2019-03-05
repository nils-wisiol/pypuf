"""
Plots to visualize results by experiments or studies.
"""
import matplotlib.pyplot as plt
from itertools import cycle
from numpy import zeros


class SuccessRatePlot:
    """
    Show the ratio of experiment results with accuracy higher than a given threshold to the total number of results,
    dependent on the number of examples in the training set.
    """
    def __init__(self, filename, group_by, experiment_hashes=None, success_threshold=.7, group_labels=None):
        """
        Prepare a plot
        :param filename: destination file (PDF)
        :param results: an object with results keyed with experiment ids
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
        self.figure.set_size_inches(w=3.34, h=1.7)
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
            legend = self.axis.legend(loc=2, fontsize=self.legend_size)
            self.figure.savefig(self.filename, bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0)


class PermutationIndexPlot:
    def __init__(self, filename, results, group_by, experiment_ids=None, group_labels=None, group_subplot_layout=None):
        self.title_size = 4
        self.tick_size = 3
        self.x_label_size = 4
        self.legend_size = 3

        self.n = None
        self.k = None
        self.experiment = None
        self.results = None

        self.filename = filename
        self.results = results
        self.experiment_ids = experiment_ids or []
        self.group_by = group_by
        self.group_labels = {} if group_labels is None else group_labels

        self.figure = plt.figure()
        self.figure.set_size_inches(w=3.34, h=1.7)
        self.group_subplot_layout = group_subplot_layout

        self.plot_data = None

    def plot(self):
        self.figure.clear()
        results = []
        if self.experiment_ids:
            for experiment_id in self.experiment_ids:
                result = self.results[experiment_id]
                if result and isinstance(result, ExperimentResult):
                    results.append(result)
        else:
            results = [r for r in self.results.values() if r]

        if len(results) == 0:
            return

        groups = sorted(set([str(getattr(r, self.group_by)) for r in results]))
        print(groups)
        assert len(set([r.experiment for r in results])) == 1

        if self.group_subplot_layout:
            group_subplot_layout = self.group_subplot_layout
        else:
            group_subplot_layout = {groups[i]: (len(groups), 1, i + 1) for i in range(len(groups))}

        print(group_subplot_layout)
        assert all(group in group_subplot_layout for group in groups), "Subplot data must be given for all groups."

        self.experiment = results[0].experiment

        axis, legend = None, None
        for i, group in enumerate(groups):
            axis = self.figure.add_subplot(*group_subplot_layout[group])
            axis.tick_params(width=0.25, which='both', labelsize=self.tick_size, direction='in')
            for a in ['top', 'bottom', 'left', 'right']:
                axis.spines[a].set_linewidth(0.5)
            axis.set_title(self.group_labels[group] if group in self.group_labels else group, size=self.title_size)
            group_results = [r for r in results if str(getattr(r, self.group_by)) == group]
            permutation_indices = [
                r.best_iteration + 1 for r in group_results if r.best_iteration >= 0
            ]
            if len(permutation_indices) == 0:
                continue
            permutation_indices.sort()
            print(permutation_indices)
            m = max(permutation_indices)
            amounts, _, _ = axis.hist(permutation_indices, density=True, label='Rel. Frequency', bins=range(1, m + 2),
                                      histtype='step', linestyle='--', linewidth=0.5)
            xs = [1]
            ys = [0]
            s = 0
            for idx, amount in enumerate(amounts):
                s += amount
                xs.append(idx + 2)
                ys.append(s)
            axis.plot(xs, ys, label='Cum. Rel. Frequency', linewidth=0.5)
            if i == len(groups) // 2:
                axis.legend(loc=7, fontsize=self.legend_size)

        plt.tight_layout()
        self.figure.savefig(self.filename)
