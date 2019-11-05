"""
This module describes a study that defines a set of experiments in order to find good hyperparameters and to examine
the quality of Deep Learning based modeling attacks on XOR Arbiter PUFs. Furthermore, some plots are defined to
visualize the experiment's results.
The Deep Learning technique used here is the Optimization technique Adam for a Stochastic Gradient Descent on a Feed-
Forward Neural Network architecture called Multilayer Perceptron (MLP). Implementations of the MLP and Adam are
used from Scikit-Learn and Tensorflow, respectively.
"""
from seaborn import catplot

from pypuf.studies.base import Study
from pypuf.studies.mlp.aseeri import Parameters, ExperimentMLPScikitLearn


class MLPDiversePUFsStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """

    COMBINER = 'xor'
    ACTIVATION = 'relu'
    ITERATION_LIMIT = 100
    LOSS = 'log_loss'
    DOMAIN_IN = -1
    PATIENCE = 30
    TOLERANCE = 0.0025
    PENALTY = 0.0002
    BETA_1 = 0.9
    BETA_2 = 0.999
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False
    SHUFFLE = True

    EXPERIMENTS = []

    TRANSFORMATIONS = ['atf', 'lightweight_secure', 'fixed_permutation']

    PREPROCESSINGS = ['full']  # ['no', 'short', 'full']

    SIZES = {
        (64, 4): [0.4e6],
        (64, 5): [0.8e6],
        (64, 6): [2e6],
        (64, 7): [5e6],
        (64, 8): [30e6],
    }

    SAMPLES_PER_POINT = {
        (64, 4): 5,
        (64, 5): 5,
        (64, 6): 5,
        (64, 7): 5,
        (64, 8): 5,
    }

    LEARNING_RATES = {
        (64, 4): [0.0025],
        (64, 5): [0.0025],
        (64, 6): [0.0055],
        (64, 7): [0.002],
        (64, 8): [0.001],
    }

    LAYERS = {
        (64, 4): [[2 ** 4, 2 ** 4, 2 ** 4]],
        (64, 5): [[2 ** 5, 2 ** 5, 2 ** 5]],
        (64, 6): [[2 ** 6, 2 ** 6, 2 ** 6]],
        (64, 7): [[2 ** 7, 2 ** 7, 2 ** 7]],
        (64, 8): [[2 ** 8, 2 ** 8, 2 ** 8]],
    }

    BATCH_SIZES = {
        (64, 4): [2**5, 2**8, 2**11],
        (64, 5): [2**5, 2**8, 2**11],
        (64, 6): [2**5, 2**8, 2**11, 2**14],
        (64, 7): [2**5, 2**8, 2**11, 2**14],
        (64, 8): [2**5, 2**8, 2**11, 2**14],
    }

    def experiments(self):
        """
        Generate an experiment for every parameter combination corresponding to the definitions above.
        For each combination of (size, samples_per_point, learning_rate) different random seeds are used.
        """
        for c1, (n, k) in enumerate(self.SIZES.keys()):
            for N in self.SIZES[(n, k)]:
                validation_frac = max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N
                for c2 in range(self.SAMPLES_PER_POINT[(n, k)]):
                    for c3, learning_rate in enumerate(self.LEARNING_RATES[(n, k)]):
                        cycle = c1 * (self.SAMPLES_PER_POINT[(n, k)] * len(self.LEARNING_RATES[(n, k)])) \
                                + c2 * len(self.LEARNING_RATES[(n, k)]) + c3
                        for preprocessing in self.PREPROCESSINGS:
                            for layers in self.LAYERS[(n, k)]:
                                for transformation in self.TRANSFORMATIONS:
                                    for batch_size in self.BATCH_SIZES[(n, k)]:
                                        self.EXPERIMENTS.append(
                                            ExperimentMLPScikitLearn(
                                                progress_log_prefix=self.name(),
                                                parameters=Parameters(
                                                    seed_simulation=0x3 + cycle,
                                                    seed_challenges=0x1415 + cycle,
                                                    seed_model=0x9265 + cycle,
                                                    seed_distance=0x3589 + cycle,
                                                    n=n,
                                                    k=k,
                                                    N=int(N),
                                                    validation_frac=validation_frac,
                                                    transformation=transformation,
                                                    combiner=self.COMBINER,
                                                    preprocessing=preprocessing,
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
                                                    batch_size=batch_size,
                                                    print_learning=self.PRINT_LEARNING,
                                                    domain_out=-1,
                                                    loss='log_loss',
                                                )
                                            )
                                        )
        return self.EXPERIMENTS

    def plot(self):
        data = self.experimenter.results
        data['Ne6'] = data.apply(lambda row: row['N'] / 10e6, axis=1)
        f = catplot(
            data=data,
            x='Ne6',
            y='accuracy',
            col='k',
            row='transformation',
            hue='layers',
            aspect=.4,
            height=10,
        )
        f.fig.savefig(f'figures/{self.name()}.pdf')
