"""
This module describes a study that defines a set of experiments in order to find good hyperparameters and to examine
the quality of Deep Learning based modeling attacks on XOR Arbiter PUFs. Furthermore, some plots are defined to
visualize the experiment's results.
The Deep Learning technique used here is the Optimization technique Adam for a Stochastic Gradient Descent on a Feed-
Forward Neural Network architecture called Multilayer Perceptron (MLP). Implementations of the MLP and Adam are
used from Scikit-Learn and Tensorflow, respectively.
"""

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
    MIN_NUM_VAL = 20
    PRINT_LEARNING = False

    PLOT_ESTIMATORS = []

    SIZES = {
        (32, 1): [4e3],
        (32, 2): [8e3],
        (32, 3): [15e3],
        (64, 1): [5e3],
        (64, 2): [10e3],
        (64, 3): [25e3],
        (128, 1): [2e3],
        (128, 2): [10e3],
        (128, 3): [125e3],
        (256, 1): [4e3],
        (256, 2): [20e3],
        (256, 3): [250e3],
        (512, 1): [8e3],
        (512, 2): [40e3],
        (512, 3): [500e3],
        (1024, 1): [16e3],
        (1024, 2): [80e3],
        (1024, 3): [1000e3],
    }

    SAMPLES_PER_POINT = {
        (32, 1): 40,
        (32, 2): 40,
        (32, 3): 40,
        (64, 1): 40,
        (64, 2): 40,
        (64, 3): 40,
        (128, 1): 40,
        (128, 2): 40,
        (128, 3): 40,
        (256, 1): 40,
        (256, 2): 40,
        (256, 3): 40,
        (512, 1): 40,
        (512, 2): 40,
        (512, 3): 40,
        (1024, 1): 40,
        (1024, 2): 40,
        (1024, 3): 40,
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
        (32, 1): [0.037, 0.0375, 0.038, 0.0385, 0.039, 0.04, 0.045, 0.041, 0.0415, 0.042, 0.0425, 0.043, 0.0435, 0.044, 0.0445, 0.045, 0.0455, 0.046, 0.0465, 0.047, 0.0475, 0.048, 0.0485, 0.049, 0.0495, 0.05],
        (32, 2): [0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08],
        (32, 3): [0.017, 0.0172, 0.0174, 0.0176, 0.0178, 0.018, 0.0182, 0.0184, 0.0186, 0.0188, 0.019, 0.0192, 0.0194 ,0.0196, 0.0198, 0.02, 0.0202, 0.0204, 0.0206, 0.0208, 0.021, 0.0212, 0.0214, 0.0216, 0.0218, 0.022, 0.0222, 0.0224, 0.0226, 0.0228, 0.023, 0.0232, 0.0234, 0.0236, 0.0238, 0.024, 0.0242, 0.0244, 0.0246, 0.0248, 0.025, 0.0252, 0.0254, 0.0256, 0.0258, 0.026, 0.0263, 0.0267, 0.027, 0.0273, 0.0277, 0.028, 0.0283, 0.0287, 0.029, 0.0293, 0.0297, 0.03],
        (64, 1): [0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071,0.072, 0.073, 0.074, 0.075, 0.085, 0.0875, 0.09, 0.0925, 0.095, 0.0975, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3],
        (64, 2): [0.0225, 0.023, 0.0235, 0.024, 0.0245, 0.025, 0.0255, 0.026, 0.0265, 0.027, 0.0275, 0.028, 0.0285, 0.029, 0.0295, 0.03, 0.0305, 0.031, 0.0315, 0.032, 0.0325, 0.033],
        (64, 3): [0.017, 0.0172, 0.0174, 0.0176, 0.0178, 0.018, 0.0182, 0.0184, 0.0186, 0.0188, 0.019, 0.0192, 0.0194 ,0.0196, 0.0198, 0.02, 0.0202, 0.0204, 0.0206, 0.0208, 0.021, 0.0212, 0.0214, 0.0216, 0.0218, 0.022, 0.0222, 0.0224, 0.0226, 0.0228, 0.023, 0.0232, 0.0234, 0.0236, 0.0238, 0.024, 0.0242, 0.0244, 0.0246, 0.0248, 0.025, 0.0252, 0.0254, 0.0256, 0.0258, 0.026, 0.0263, 0.0267, 0.027, 0.0273, 0.0277, 0.028, 0.0283, 0.0287, 0.029, 0.0293, 0.0297, 0.03],
        (128, 1): [0.04, 0.0402, 0.0404, 0.0406, 0.0408, 0.041, 0.0412, 0.0414, 0.0416, 0.0418, 0.042, 0.0422, 0.0424, 0.0426, 0.0428, 0.043, 0.0432, 0.0434, 0.0436, 0.0438, 0.044, 0.0442, 0.0444, 0.0446, 0.0448, 0.045, 0.0452, 0.0454, 0.0456, 0.0458, 0.046, 0.0462, 0.0464, 0.0466, 0.0468, 0.047, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095],
        (128, 2): [0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.015, 0.0155, 0.016, 0.0165, 0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.0205, 0.021, 0.0215, 0.022, 0.0225, 0.023, 0.0235, 0.024, 0.0245, 0.025, 0.0255, 0.026, 0.0265, 0.027, 0.0275, 0.028, 0.0285, 0.029, 0.0295, 0.03],
        (128, 3): [0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.015, 0.0155, 0.016, 0.0165, 0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.0205, 0.021, 0.0215, 0.022, 0.0225, 0.023, 0.0235, 0.024, 0.0245, 0.025, 0.0255, 0.026, 0.0265, 0.027, 0.0275, 0.028, 0.0285, 0.029, 0.0295, 0.03],
        (256, 1): [0.05, 0.0505, 0.051, 0.0515, 0.052, 0.0525, 0.053, 0.0535, 0.054, 0.0545, 0.055, 0.0555, 0.056, 0.0565, 0.057, 0.0575, 0.058, 0.0585, 0.059, 0.0595, 0.06, 0.0605, 0.061, 0.0615, 0.062, 0.0625, 0.063, 0.0635, 0.064, 0.0645, 0.065, 0.0655, 0.066, 0.0665, 0.067, 0.0675, 0.068, 0.0685, 0.069, 0.0695, 0.07],
        (256, 2): [0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.015, 0.0155, 0.016, 0.0165, 0.017, 0.0175],
        (256, 3): [0.01, 0.0103, 0.0107, 0.011, 0.0113, 0.0117, 0.012, 0.0123, 0.0127, 0.013, 0.0133, 0.0137, 0.014, 0.0143, 0.0147, 0.015, 0.0153, 0.0157, 0.016, 0.0163, 0.0167, 0.017, 0.0173, 0.0177, 0.018],
        (512, 1): [0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09],
        (512, 2): [0.0025, 0.00275, 0.003, 0.00325, 0.0035, 0.00375, 0.004, 0.00425, 0.0045, 0.00475, 0.005, 0.00525, 0.0055, 0.00575, 0.006, 0.00625, 0.0065, 0.00675, 0.007, 0.00725, 0.0075, 0.00775, 0.008],
        (512, 3): [0.0025, 0.00275, 0.003, 0.00325, 0.0035, 0.00375, 0.004, 0.00425, 0.0045, 0.00475, 0.005, 0.00525, 0.0055, 0.00575, 0.006, 0.00625, 0.0065, 0.00675, 0.007, 0.00725, 0.0075, 0.00775, 0.008, 0.00825, 0.0085, 0.00875, 0.009, 0.00925, 0.0095, 0.00975, 0.01, 0.01025, 0.0105, 0.01075, 0.011, 0.01125, 0.0115, 0.01175, 0.012, 0.01225, 0.0125, 0.01275, 0.013, 0.01325, 0.0135, 0.01375, 0.014],
        (1024, 1): [0.0175, 0.01875, 0.02, 0.02125, 0.0225, 0.02375, 0.025, 0.02625, 0.0275, 0.02875, 0.03, 0.03125, 0.0325, 0.03375, 0.035, 0.03625, 0.0375, 0.03875, 0.04, 0.04125, 0.0425, 0.04375, 0.045, 0.04625, 0.0475, 0.04875, 0.05, 0.05125, 0.0525, 0.05375, 0.055],
        (1024, 2): [0.0025, 0.00275, 0.003, 0.00325, 0.0035, 0.00375, 0.004, 0.00425, 0.0045, 0.00475, 0.005, 0.00525, 0.0055, 0.00575, 0.006, 0.00625, 0.0065, 0.00675, 0.007, 0.00725, 0.0075, 0.00775, 0.008, 0.00825, 0.0085, 0.00875, 0.009, 0.00925, 0.0095, 0.00975, 0.01, 0.01025, 0.0105, 0.01075, 0.011, 0.01125, 0.0115, 0.01175, 0.012, 0.01225, 0.0125, 0.01275, 0.013, 0.01325, 0.0135, 0.01375, 0.014],
        (1024, 3): [0.0025, 0.00275, 0.003, 0.00325, 0.0035, 0.00375, 0.004, 0.00425, 0.0045, 0.00475, 0.005, 0.00525, 0.0055, 0.00575, 0.006, 0.00625, 0.0065, 0.00675, 0.007, 0.00725, 0.0075, 0.00775, 0.008],
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
        if not self.EXPERIMENTS:
            self.experiments()
        self.plot_helper(
            name='ScikitLearn',
            df=self.experimenter.results,
            param_x='learning_rate',
            param_1='N',
            param_2=None,
        )

    def plot_helper(self, name, df, param_x, param_1, param_2=None):
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

