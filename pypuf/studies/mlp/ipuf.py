"""
This module describes a study that defines a set of experiments in order to find good hyperparameters and to examine
the quality of Deep Learning based modeling attacks on XOR Arbiter PUFs. Furthermore, some plots are defined to
visualize the experiment's results.
The Deep Learning technique used here is the Optimization technique Adam for a Stochastic Gradient Descent on a Feed-
Forward Neural Network architecture called Multilayer Perceptron (MLP). Implementations of the MLP and Adam are
used from Scikit-Learn and Tensorflow, respectively.
"""
import re

from matplotlib.pyplot import close
from numpy import isnan
from seaborn import catplot, axes_style

from pypuf.experiments.experiment.mlp_skl import ExperimentMLPScikitLearn, Parameters as Parameters_skl
from pypuf.studies.base import Study


class InterposeMLPStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """
    SHUFFLE = True

    ITERATION_LIMIT = 50
    PATIENCE = ITERATION_LIMIT
    TOLERANCE = 0.0025
    PENALTY = 0.0002
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False

    def experiments(self):
        return [
            ExperimentMLPScikitLearn(
                progress_log_prefix=self.name(),
                parameters=Parameters_skl(
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
            for seed in range(10)
            for batch_size in [10**5]
            for layer_size in [128, 192, 256, 386, 512]
            for n in [64]
            for (k_up, k_down), (N_set, LR_set) in {
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
            }.items()
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
        data['max_memory_gb'] = data.apply(lambda row: row['max_memory'] / 1024**3, axis=1)
        data['Ne6'] = data.apply(lambda row: row['N'] / 1e6, axis=1)
        data['size'] = data.apply(lambda row: get_size(row), axis=1)
        data = data.sort_values(['size', 'layers'])
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
