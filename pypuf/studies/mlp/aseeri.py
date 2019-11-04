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
from os import getpid
from typing import NamedTuple, Iterable
from uuid import UUID
from uuid import uuid4

from matplotlib.pyplot import subplots
from numpy.ma import ones
from numpy.random.mtrand import RandomState
from numpy.random.mtrand import seed
from pandas import DataFrame
from seaborn import stripplot, boxplot

from pypuf import tools
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.neural_networks.mlp_skl import MultiLayerPerceptronScikitLearn
from pypuf.learner.neural_networks.mlp_tf import MultiLayerPerceptronTensorflow
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF
from pypuf.studies.base import Study


class Parameters(NamedTuple):
    """
    Define all input parameters for the Experiment.
    """
    seed_simulation: int
    seed_challenges: int
    seed_model: int
    seed_distance: int
    n: int
    k: int
    N: int
    validation_frac: float
    transformation: str
    combiner: str
    preprocessing: str
    layers: Iterable[int]
    activation: str
    loss: str
    domain_in: int
    domain_out: int
    learning_rate: float
    penalty: float
    beta_1: float
    beta_2: float
    tolerance: float
    patience: int
    iteration_limit: int
    batch_size: int
    print_learning: bool


class Result(NamedTuple):
    """
    Define all parameters to be documented within the result file that are not included in the input parameters.
    """
    name: str
    experiment_id: UUID
    pid: int
    measured_time: float
    iterations: int
    accuracy: float
    loss_curve: Iterable[float]
    accuracy_curve: Iterable[float]
    max_memory: int


class ExperimentMLPTensorflow(Experiment):
    """
    This Experiment uses the Tensorflow implementation of the Multilayer Perceptron Learner.
    """

    NAME = 'Multilayer Perceptron (tensorflow)'
    NUM_ACCURACY = 10000

    def __init__(self, progress_log_prefix, parameters):
        self.id = uuid4()
        progress_log_name = None if not progress_log_prefix else \
            '{}_{}'.format(progress_log_prefix, self.id)
        super().__init__(progress_log_name=progress_log_name, parameters=parameters)
        self.training_set = None
        self.simulation = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.progress_logger.debug('Setting up simulation')
        self.simulation = XORArbiterPUF(
            n=self.parameters.n,
            k=self.parameters.k,
            seed=self.parameters.seed_simulation,
            transform=self.parameters.transformation,
        )
        self.progress_logger.debug(f'Gathering training and test set with {self.parameters.N} examples')
        prng_challenges = RandomState(seed=self.parameters.seed_challenges)
        N_val = int(self.parameters.N * self.parameters.validation_frac)
        N_train = self.parameters.N - N_val
        self.training_set = tools.TrainingSet(instance=self.simulation, N=N_train, random_instance=prng_challenges)
        validation_set = tools.TrainingSet(instance=self.simulation, N=N_val, random_instance=prng_challenges)
        self.learner = MultiLayerPerceptronTensorflow(
            n=self.parameters.n,
            k=self.parameters.k,
            training_set=self.training_set,
            validation_set=validation_set,
            transformation=self.simulation.transform,
            preprocessing=self.parameters.preprocessing,
            layers=self.parameters.layers,
            activation=self.parameters.activation,
            loss=self.parameters.loss,
            domain_in=self.parameters.domain_in,
            domain_out=self.parameters.domain_out,
            learning_rate=self.parameters.learning_rate,
            penalty=self.parameters.penalty,
            beta_1=self.parameters.beta_1,
            beta_2=self.parameters.beta_2,
            tolerance=self.parameters.tolerance,
            patience=self.parameters.patience,
            iteration_limit=self.parameters.iteration_limit,
            batch_size=self.parameters.batch_size,
            seed_model=self.parameters.seed_model,
            print_learning=self.parameters.print_learning,
        )
        self.learner.prepare()

    def run(self):
        """
        Execute the learning process.
        """
        self.model = self.learner.learn()

    def analyze(self):
        """
        Calculate statistics of the trained model and write them into a result log file.
        """
        assert self.model is not None
        accuracy = 1.0 - tools.approx_dist(
                instance1=self.simulation,
                instance2=self.model,
                num=min(self.NUM_ACCURACY, 2 ** self.parameters.n),
                random_instance=RandomState(seed=self.parameters.seed_distance),
        )
        return Result(
            name=self.NAME,
            experiment_id=self.id,
            pid=getpid(),
            accuracy=accuracy,
            iterations=self.learner.history.epoch[-1],
            measured_time=self.measured_time,
            loss_curve=[round(loss, 3) for loss in self.learner.history.history['val_loss']],
            accuracy_curve=[round(accuracy, 3) for accuracy in self.learner.history.history['val_pypuf_accuracy']],
            max_memory=self.max_memory(),
        )


class ExperimentMLPScikitLearn(Experiment):
    """
    This Experiment uses the Scikit-learn implementation of the Multilayer Perceptron Learner.
    """

    NAME = 'Multilayer Perceptron (scikit-learn)'

    def __init__(self, progress_log_prefix, parameters):
        self.id = uuid4()
        progress_log_name = None if not progress_log_prefix else \
            '{}_{}'.format(progress_log_prefix, self.id)
        super().__init__(progress_log_name=progress_log_name, parameters=parameters)
        self.training_set = None
        self.simulation = None
        self.learner = None
        self.model = None
        assert self.parameters.domain_out == -1, 'domain_out other than -1 is not supported, sorry.'
        assert self.parameters.loss == 'log_loss', 'loss other than log_loss is not supported, sorry.'

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.progress_logger.debug('Setting up simulation')
        self.simulation = XORArbiterPUF(
            n=self.parameters.n,
            k=self.parameters.k,
            seed=self.parameters.seed_simulation,
            transform=self.parameters.transformation,
        )
        self.progress_logger.debug(f'Gathering training set with {self.parameters.N} examples')
        self.training_set = tools.TrainingSet(
            instance=self.simulation,
            N=self.parameters.N,
            random_instance=RandomState(seed=self.parameters.seed_challenges),
        )
        self.progress_logger.debug('Setting up learner')
        self.learner = MultiLayerPerceptronScikitLearn(
            n=self.parameters.n,
            k=self.parameters.k,
            training_set=self.training_set,
            validation_frac=self.parameters.validation_frac,
            transformation=self.simulation.transform,
            preprocessing=self.parameters.preprocessing,
            layers=self.parameters.layers,
            learning_rate=self.parameters.learning_rate,
            penalty=self.parameters.penalty,
            beta_1=self.parameters.beta_1,
            beta_2=self.parameters.beta_2,
            tolerance=self.parameters.tolerance,
            patience=self.parameters.patience,
            iteration_limit=self.parameters.iteration_limit,
            batch_size=self.parameters.batch_size,
            seed_model=self.parameters.seed_model,
            print_learning=self.parameters.print_learning,
            logger=self.progress_logger.debug,
        )
        self.learner.prepare()

    def run(self):
        """
        Execute the learning process.
        """
        self.progress_logger.debug('Starting learner')
        self.model = self.learner.learn()

    def analyze(self):
        """
        Calculate statistics of the trained model and write them into a result log file.
        """
        self.progress_logger.debug('Analyzing result')
        assert self.model is not None
        accuracy = 1.0 - tools.approx_dist(
            instance1=self.simulation,
            instance2=self.model,
            num=10**4,
            random_instance=RandomState(seed=self.parameters.seed_distance),
        )
        return Result(
            name=self.NAME,
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            iterations=self.learner.nn.n_iter_,
            accuracy=accuracy,
            loss_curve=[round(loss, 3) for loss in self.learner.nn.loss_curve_],
            accuracy_curve=[round(accuracy, 3) for accuracy in self.learner.accuracy_curve],
            max_memory=self.max_memory(),
        )


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

    EXPERIMENTER_CALLBACK_MIN_PAUSE = 60

    PLOT_ESTIMATORS = []

    SIZES = [
        (64, 4, 0.4e6),
        (64, 5, 0.8e6),
        # (64, 6, 2e6),
        # (64, 7, 5e6),
        # (64, 8, 30e6),
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

    REFERENCE = DataFrame(
        columns=['k', 'accuracy', 'measured_time'],
        data=[
            [4, .9842, 19.2],
            [5, .9855, 58.0],
            [6, .9915, 7.4 * 60],
            [7, .9921, 11.8 * 60],
            [8, .9874, 23.3 * 60],
        ],
    )
    REFERENCE['distance'] = 1 - REFERENCE['accuracy']

    EXPERIMENTS = []

    NICE_EXPERIMENT_NAMES = {
        'ExperimentMLPScikitLearn': 'Scikit-Learn',
        'ExperimentMLPTensorflow': 'Tensorflow',
    }

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
                                        loss='log_loss',
                                        domain_out=-1,
                                    )
                                )
                            )
                            for loss in self.LOSSES[k]:
                                self.EXPERIMENTS.append(
                                    ExperimentMLPTensorflow(
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
        # set up data
        data = self.experimenter.results
        ncols = data['experiment'].nunique()
        data['distance'] = 1 - data['accuracy']

        # set up figure
        seed(42)
        nrows = 2
        fig, axes = subplots(ncols=ncols, nrows=2)
        fig.set_size_inches(7 * ncols, 4 * nrows)
        axes = axes.reshape((nrows, ncols))

        # plot
        for i, (experiment, experiment_data) in enumerate(data.groupby(['experiment'])):
            # accuracy for this experiment
            self._figure('distance', 'accuracy', '{:.4f}', experiment_data, axes[0][i])
            axes[0][i].invert_yaxis()
            axes[0][i].set_xlim(left=-0.5, right=4.485)
            axes[0][i].set_ylim(top=0.005, bottom=0.6)
            major_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
            minor_ticks = [0.01, 0.02, 0.03, 0.04, 0.05]
            axes[0][i].set_yticks(ticks=major_ticks, minor=False)
            axes[0][i].set_yticks(ticks=minor_ticks, minor=True)
            axes[0][i].set_yticklabels(ones(shape=5) - major_ticks, minor=False)
            axes[0][i].set_yticklabels(ones(shape=5) - minor_ticks, minor=True)
            axes[0][i].grid(b=True, which='minor', color='gray', linestyle='--')
            axes[0][i].set_ylabel('accuracy')
            axes[0][i].set_title(f'Library: {self.NICE_EXPERIMENT_NAMES.get(experiment, experiment)}')

            # run time for this experiment
            self._figure('measured_time', 'measured_time', '{:.0f}', experiment_data, axes[1][i])
            axes[1][i].set_ylabel('runtime in s')
            axes[1][i].set_title(f'Library: {self.NICE_EXPERIMENT_NAMES.get(experiment, experiment)}')

        # write out
        fig.subplots_adjust(wspace=0.6)
        fig.subplots_adjust(hspace=0.3, wspace=0.6)
        title = fig.suptitle('Accuracy and Run Time of MLP Learning 64-bit k-XOR Arbiter PUFs\n'
                             '(Reference Values are Shown in Black)', size=20)
        title.set_position([0.5, 1.05])
        fig.savefig(f'figures/{self.name()}.png', bbox_inches='tight', pad_inches=.5)
        fig.savefig(f'figures/{self.name()}.pdf', bbox_inches='tight', pad_inches=.5)

    def _figure(self, y, y_legend, legend_format, data, ax):
        stripplot(
            x='k',
            y=y,
            data=data,
            ax=ax,
            jitter=True,
            alpha=0.4,
            zorder=1,
        )
        ax.legend(
            list(map(lambda f: legend_format.format(f), data.sort_values(['k']).groupby(['k']).mean()[y_legend])),
            loc='upper right', bbox_to_anchor=(1.26, 1.02), title='means',
        )
        boxplot(
            x='k',
            y=y,
            data=self.REFERENCE,
            ax=ax,
        )
        ax.set_yscale('log')
