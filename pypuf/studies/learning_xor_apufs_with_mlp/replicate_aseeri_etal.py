"""
This module describes a study that defines a set of experiments in order to replicate the results from Aseeri et al. [1]
of Deep Learning based modeling attacks on (k, 64)-XOR Arbiter PUFs. Furthermore, some plots are defined to
visualize the experiment's results.
The Deep Learning technique used here is the Optimization technique Adam [2] for a Stochastic Gradient Descent on a
Feed-Forward Neural Network architecture called Multilayer Perceptron (MLP) that is an extension to the Perceptron [3].
Implementations of the MLP and Adam are used each from both Scikit-Learn [4] and Tensorflow [5].

References:
[1]  A. Aseeri et al.,      "A Machine Learning-based Security Vulnerability Study on XOR PUFs for Resource-Constraint
                            Internet of Things", IEEE International Congress on Internet of Things (ICIOT),
                            San Francisco, CA, pp. 49-56, 2018.
[2]  D. Kingma and J. Ba,   “Adam: A Method for Stochastic Optimization”, arXiv:1412.6980, 2014.
[3]  F. Rosenblatt,         "The Perceptron: A Probabilistic Model for Information Storage and Organization in the
                            Brain.", Psychological Review, volume 65, pp. 386-408, 1958.
[4]  F., Pedregosa et al.,  "Scikit-learn: Machine learning in Python", Journal of Machine Learning Research, volume 12,
                            pp. 2825-2830, 2011.
                            https://scikit-learn.org
[5]  M. Abadi et al.,       "Tensorflow: A System for Large-scale Machine Learning", 12th USENIX Symposium on Operating
                            Systems Design and Implementation, pp. 265–283, 2016.
                            http://tensorflow.org/
"""
from itertools import product

from matplotlib.pyplot import subplots
from numpy.ma import ones
from numpy.random.mtrand import seed
from seaborn import stripplot

from pypuf.studies.base import Study
from pypuf.experiments.experiment.mlp_tf import ExperimentMLPTensorflow, Parameters as Parameters_tf
from pypuf.experiments.experiment.mlp_skl import ExperimentMLPScikitLearn, Parameters as Parameters_skl


class ReplicateAseeriEtAlStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """

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
        4: 100,
        5: 100,
        6: 100,
        7: 100,
        8: 100,
    }

    LAYERS = {
        4: [[2 ** 4, 2 ** 4, 2 ** 4]],
        5: [[2 ** 5, 2 ** 5, 2 ** 5]],
        6: [[2 ** 6, 2 ** 6, 2 ** 6]],
        7: [[2 ** 7, 2 ** 7, 2 ** 7]],
        8: [[2 ** 8, 2 ** 8, 2 ** 8]],
    }

    LOSSES = {
        4: ['log_loss'],
        5: ['log_loss'],
        6: ['log_loss'],
        7: ['log_loss'],
        8: ['log_loss'],
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
        4: [0.0025],
        5: [0.0025],
        6: [0.0055],
        7: [0.002],
        8: [0.001],
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
    }

    REFERENCE_MEAN_ACCURACY = {
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
        Different random seeds are used for each combination of (size, samples_per_point, learning_rate).
        """
        for c1, (n, k, N) in enumerate(self.SIZES):
            for c2 in range(self.SAMPLES_PER_POINT[k]):
                for c3, learning_rate in enumerate(self.LEARNING_RATES[k]):
                    cycle = c1 * (self.SAMPLES_PER_POINT[k] * len(self.LEARNING_RATES[k])) \
                            + c2 * len(self.LEARNING_RATES[k]) + c3
                    for domain_in, domain_out in self.DOMAINS[k]:
                        validation_frac = max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N
                        for (layers, penalty, beta_1, beta_2, patience, tolerance) in list(product(*[
                            self.LAYERS[k], self.PENALTIES[k], self.BETAS_1[k], self.BETAS_2[k], self.PATIENCE[k],
                            self.TOLERANCES[k]
                        ])):
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
        """
        Visualize the quality of learning by plotting the accuracy of each experiment grouped by k,
        plotting the mean value for each group, and that from Aseeri et. al. in black, respectively.
        """
        if not self.EXPERIMENTS:
            self.experiments()
        seed(42)
        df = self.experimenter.results
        ncols = 2
        nrows = 2
        fig, axes = subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(7 * ncols, 4 * nrows)
        axes = axes.reshape((nrows, ncols))
        distances = 1 - df['accuracy']
        df['distance'] = distances
        color_ref = 'black'
        style_ref = '-'
        marker = 'o'
        width = 0.5

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
            for j, accuracy_ref in enumerate(self.REFERENCE_MEAN_ACCURACY.values()):
                axes[0][i].plot((-0.25 + j, 0.235 + j), 2 * (means_accuracy[j],),
                                linewidth=2, label=str(round(means_accuracy[j], 4)))
                axes[0][i].plot((-0.25 + j, 0.235 + j), 2 * (accuracy_ref,),
                                color=color_ref, linestyle=style_ref, linewidth=2, zorder=2)
            lib = 'Tensorflow' if 'Tensorflow' in experiment \
                else 'Scikit-learn' if 'ScikitLearn' in experiment else '?'
            axes[0][i].set_xlim(left=-0.5, right=4.485)
            axes[0][i].set_title('Library: {}\n'.format(lib))
            axes[0][i].set_yscale('linear')
            axes[0][i].set_ylabel('accuracy')
            axes[0][i].legend(loc='upper right', bbox_to_anchor=(1.26, 1.02), title='means')

            stripplot(
                x='k',
                y='distance',
                data=df[df['experiment'] == experiment],
                ax=axes[1][i],
                jitter=True,
                alpha=0.4,
                zorder=1,
                marker=marker,
            )
            means_distance = [df[(df.experiment == experiment) & (df.k == k)]['distance'].mean()
                              for k in sorted(list(set(df['k'])))]
            for j, accuracy_ref in enumerate(self.REFERENCE_MEAN_ACCURACY.values()):
                axes[1][i].plot((-0.25 + j, 0.235 + j), 2 * (means_distance[j],),
                                linewidth=2, label=str(round(means_distance[j], 3)))
                axes[1][i].plot((-0.25 + j, 0.235 + j), 2 * (1 - accuracy_ref,),
                                color=color_ref, linestyle=style_ref, linewidth=2, zorder=2)
            axes[1][i].set_yscale('log')
            axes[1][i].invert_yaxis()
            axes[1][i].set_xlim(left=-0.5, right=4.485)
            axes[1][i].set_ylim(top=0.005, bottom=0.6)
            major_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
            minor_ticks = [0.01, 0.02, 0.03, 0.04, 0.05]
            axes[1][i].set_yticks(ticks=major_ticks, minor=False)
            axes[1][i].set_yticks(ticks=minor_ticks, minor=True)
            axes[1][i].set_yticklabels(ones(shape=5) - major_ticks, minor=False)
            axes[1][i].set_yticklabels(ones(shape=5) - minor_ticks, minor=True)
            axes[1][i].grid(b=True, which='minor', color='gray', linestyle='--')
            axes[1][i].set_ylabel('accuracy')
        fig.subplots_adjust(hspace=0.3, wspace=0.6)
        title = fig.suptitle('Accuracy Overview', size=20)
        title.set_position([0.5, 1.05])
        fig.savefig('figures/{}_overview_accuracy.png'.format(self.name()), bbox_inches='tight', pad_inches=.5)

        fig2, axes2 = subplots(ncols=2, nrows=1)
        fig2.set_size_inches(14, 4)
        color_ref = 'black'
        style_ref = '-'
        marker = 'o'
        for i, experiment in enumerate(sorted(list(set(df['experiment'])))):
            stripplot(
                x='k',
                y='measured_time',
                data=df[df['experiment'] == experiment],
                ax=axes2[i],
                jitter=True,
                alpha=0.4,
                zorder=1,
                marker=marker,
            )
            means_time = [df[(df.experiment == experiment) & (df.k == k)]['measured_time'].mean()
                          for k in sorted(list(set(df['k'])))]
            for j, ref_time in enumerate(self.REFERENCE_TIMES.values()):
                axes2[i].plot((-0.25 + j, 0.235 + j), (means_time[j], means_time[j]),
                              linewidth=2, label=str(int(round(means_time[j], 0))))
                axes2[i].plot((-0.25 + j, 0.235 + j), 2 * (ref_time,),
                              color=color_ref, linestyle=style_ref, linewidth=2, zorder=2)
                if j > 0:
                    axes2[i].plot((j - 1, j), (means_time[j - 1], means_time[j]),
                                  color='gray', linestyle='-', linewidth=2, zorder=0)
            axes2[i].set_yscale('log')
            axes2[i].legend(loc='upper right', bbox_to_anchor=(1.25, 1.02), title='means')
            axes2[i].set_ylabel('runtime in s')
            lib = 'Tensorflow' if 'Tensorflow' in experiment \
                else 'Scikit-learn' if 'ScikitLearn' in experiment else '?'
            axes2[i].set_title('Library: {}\n'.format(lib))
        fig2.subplots_adjust(wspace=0.6)
        title2 = fig2.suptitle('Runtime Overview', size=20)
        title2.set_position([0.5, 1.2])
        fig2.savefig('figures/{}_overview_runtime.png'.format(self.name()), bbox_inches='tight', pad_inches=.5)
