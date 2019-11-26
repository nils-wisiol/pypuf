"""
This module describes a study that defines a set of experiments in order to examine the quality of Deep Learning based
modeling attacks on XOR Arbiter PUFs. Furthermore, some plots are defined to visualize the experiment's results.
The Deep Learning technique used here is the Feed-Forward Neural Network architecture called Multilayer Perceptron (MLP)
[1] that is applied with the optimization technique Adam [2] for a Stochastic Gradient Descent. Implementations of the
MLP and Adam are used from Scikit-Learn [3].

References:
[1]  F. Rosenblatt,         "The Perceptron: A Probabilistic Model for Information Storage and Organization in the
                            Brain.", Psychological Review, volume 65, pp. 386-408, 1958.
[2]  D. Kingma and J. Ba,   “Adam: A Method for Stochastic Optimization”, arXiv:1412.6980, 2014.
[3]  F., Pedregosa et al.,  "Scikit-learn: Machine learning in Python", Journal of Machine Learning Research, volume 12,
                            pp. 2825-2830, 2011.
                            https://scikit-learn.org
"""
import re
from os import getpid
from typing import NamedTuple, Iterable, List
from uuid import UUID
from uuid import uuid4

from matplotlib.pyplot import close
from numpy import concatenate, prod
from numpy.core._multiarray_umath import ndarray
from numpy.random.mtrand import RandomState
from seaborn import catplot, axes_style

from pypuf import tools
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.neural_networks.mlp_skl import MultiLayerPerceptronScikitLearn
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.simulation.base import Simulation
from pypuf.studies.base import Study
from pypuf.studies.ipuf.split import SplitAttackStudy


class Interpose3PUF(Simulation):

    def __init__(self, n: int, k_up: int, k_middle: int, k_down: int, seed: int) -> None:
        seeds = RandomState(seed)
        self.n = n
        self.k = k_up
        self.k_up, self.k_middle, self.k_down = k_up, k_middle, k_down
        self.up = XORArbiterPUF(n=n, k=k_up, seed=seeds.randint(0, 2 ** 32))
        self.middle = XORArbiterPUF(n=n + 1, k=k_up, seed=seeds.randint(0, 2 ** 32))
        self.down = XORArbiterPUF(n=n + 1, k=k_up, seed=seeds.randint(0, 2 ** 32))
        self.interpose_pos = n // 2

    def challenge_length(self) -> int:
        return self.up.challenge_length()

    def response_length(self) -> int:
        return self.down.response_length()

    def _interpose(self, challenges, bits):
        pos = self.interpose_pos
        return concatenate(
            (challenges[:, :pos], bits.reshape(-1, 1), challenges[:, pos:]),
            axis=1,
        )

    def eval(self, challenges: ndarray) -> ndarray:
        return self.down.eval(self._interpose(
            challenges=challenges,
            bits=self.middle.eval(self._interpose(
                challenges=challenges,
                bits=self.up.eval(challenges)
            ))
        ))


class InterposeBinaryTree(Simulation):

    def __init__(self, n: int, k_up: int, k_middle: int, k_down: int, seed: int) -> None:
        seeds = RandomState(seed)
        self.n = n
        self.k = k_up
        self.k_up, self.k_middle, self.k_down = k_up, k_middle, k_down
        self.up = XORArbiterPUF(n, k_up, seed=seeds.randint(0, 2 ** 32))
        self.middle = [XORArbiterPUF(n + 1, k_middle, seed=seeds.randint(0, 2 ** 32)) for _ in range(2)]
        self.down = [XORArbiterPUF(n + 1, k_down, seed=seeds.randint(0, 2 ** 32)) for _ in range(2)]
        self.interpose_pos = n // 2

    def challenge_length(self) -> int:
        return self.up.challenge_length()

    def response_length(self) -> int:
        return self.down[0].response_length()

    def _interpose(self, challenges, bits):
        ipos = self.interpose_pos
        return concatenate(
            challenges[:, :ipos], bits, challenges[:, ipos:],
            axis=1,
        )

    def eval(self, challenges: ndarray) -> ndarray:
        upper_responses = self.up.eval(challenges)
        return self.down[0].eval(self._interpose(
            challenges=challenges,
            bits=self.middle[0].eval(self._interpose(challenges, upper_responses))
        )) * self.down[1].eval(self._interpose(
            challenges=challenges,
            bits=self.middle[1].eval(self._interpose(challenges, upper_responses))
        ))


class InterposeCascade(Simulation):

    def __init__(self, n: int, ks: List[int], seed: int) -> None:
        seeds = RandomState(seed)
        self.n = n
        self.k = ks[0]
        self.ks = ks
        self.layers = [
            XORArbiterPUF(n=n + 1 if i > 0 else n, k=k, seed=seeds.randint(0, 2 ** 32)) for i, k in enumerate(ks)
        ]
        self.interpose_pos = n // 2

    def challenge_length(self) -> int:
        return self.layers[0].challenge_length()

    def response_length(self) -> int:
        return self.layers[-1].response_length()

    def _interpose(self, challenges, bits):
        ipos = self.interpose_pos
        return concatenate(
            challenges[:, :ipos], bits, challenges[:, ipos:],
            axis=1,
        )

    def eval(self, challenges: ndarray) -> ndarray:
        result = 1
        for i, layer in enumerate(self.layers):
            result = result * layer.eval(self._interpose(challenges=challenges, bits=result) if i > 0 else challenges)
        return result


class XORInterposePUF(Simulation):

    def __init__(self, n: int, k: int, seed: int) -> None:
        seeds = RandomState(seed)
        self.n = n
        self.k = k
        self.layers_up = [XORArbiterPUF(n=n, k=1, seed=seeds.randint(0, 2 ** 32)) for _ in range(k)]
        self.layers_down = [XORArbiterPUF(n=n + 1, k=1, seed=seeds.randint(0, 2 ** 32)) for _ in range(k)]
        self.interpose_pos = n // 2

    def challenge_length(self) -> int:
        return self.layers_up[0].challenge_length()

    def response_length(self) -> int:
        return self.layers_down[0].response_length()

    def _interpose(self, challenges, bits):
        ipos = self.interpose_pos
        return concatenate(
            challenges[:, :ipos], bits, challenges[:, ipos:],
            axis=1,
        )

    def eval(self, challenges: ndarray) -> ndarray:
        return prod([self.layers_down[i].eval(self._interpose(
            challenges=challenges,
            bits=self.layers_up[i].eval(challenges)
        )) for i in range(self.k)])


class XORInterpose3PUF(Simulation):

    def __init__(self, n: int, k: int, seed: int) -> None:
        seeds = RandomState(seed)
        self.n = n
        self.k = k
        self.layers_up = [XORArbiterPUF(n=n, k=1, seed=seeds.randint(0, 2 ** 32)) for _ in range(k)]
        self.layers_middle = [XORArbiterPUF(n=n + 1, k=1, seed=seeds.randint(0, 2 ** 32)) for _ in range(k)]
        self.layers_down = [XORArbiterPUF(n=n + 1, k=1, seed=seeds.randint(0, 2 ** 32)) for _ in range(k)]
        self.interpose_pos = n // 2

    def challenge_length(self) -> int:
        return self.layers_up[0].challenge_length()

    def response_length(self) -> int:
        return self.layers_down[0].response_length()

    def _interpose(self, challenges, bits):
        ipos = self.interpose_pos
        return concatenate(
            challenges[:, :ipos], bits, challenges[:, ipos:],
            axis=1,
        )

    def eval(self, challenges: ndarray) -> ndarray:
        return prod([self.layers_down[i].eval(self._interpose(
            challenges=challenges,
            bits=self.layers_middle[i].eval(self._interpose(
                challenges=challenges,
                bits=self.layers_up[i].eval(challenges)
            ))
        )) for i in range(self.k)])


class Parameters(NamedTuple):
    """
    Define all input parameters for the Experiment.
    """
    simulation: Simulation
    seed: int
    N: int
    validation_frac: float
    preprocessing: str
    layers: Iterable[int]
    learning_rate: float
    tolerance: float
    patience: int
    iteration_limit: int
    batch_size: int


class Result(NamedTuple):
    """
    Define all parameters to be documented within the result file that are not included in the input parameters.
    """
    name: str
    n: int
    first_k: int
    experiment_id: UUID
    pid: int
    measured_time: float
    iterations: int
    accuracy: float
    loss_curve: Iterable[float]
    accuracy_curve: Iterable[float]
    max_memory: int


class ExperimentMLPScikitLearn(Experiment):
    """
    This Experiment uses the Scikit-learn implementation of the Multilayer Perceptron Learner.
    """

    NAME = 'Multilayer Perceptron (scikit-learn)'

    def __init__(self, progress_log_prefix, parameters):
        self.id = uuid4()
        progress_log_name = None if not progress_log_prefix else f'{progress_log_prefix}_{self.id}'
        super().__init__(progress_log_name=progress_log_name, parameters=parameters)
        self.simulation = parameters.simulation
        self.training_set = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.progress_logger.debug(f'Gathering training set with {self.parameters.N} examples')
        self.training_set = tools.TrainingSet(
            instance=self.simulation,
            N=self.parameters.N,
            random_instance=RandomState(seed=self.parameters.seed),
        )
        self.progress_logger.debug('Setting up learner')
        self.learner = MultiLayerPerceptronScikitLearn(
            n=self.parameters.simulation.n,
            k=self.parameters.simulation.k,
            training_set=self.training_set,
            validation_frac=self.parameters.validation_frac,
            transformation=LTFArray.transform_atf,
            preprocessing='short',
            layers=self.parameters.layers,
            learning_rate=self.parameters.learning_rate,
            penalty=0.0002,
            beta_1=0.9,
            beta_2=0.999,
            tolerance=self.parameters.tolerance,
            patience=self.parameters.patience,
            iteration_limit=self.parameters.iteration_limit,
            batch_size=self.parameters.batch_size,
            seed_model=self.parameters.seed,
            print_learning=False,
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
            num=10 ** 4,
            random_instance=RandomState(seed=self.parameters.seed),
        )
        return Result(
            name=self.NAME,
            n=self.parameters.simulation.n,
            first_k=self.parameters.simulation.k,
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            iterations=self.learner.nn.n_iter_,
            accuracy=accuracy,
            loss_curve=[round(loss, 3) for loss in self.learner.nn.loss_curve_],
            accuracy_curve=[round(accuracy, 3) for accuracy in self.learner.accuracy_curve],
            max_memory=self.max_memory(),
        )


class InterposeMLPStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """
    SHUFFLE = True

    ITERATION_LIMIT = 200
    PATIENCE = ITERATION_LIMIT
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False
    SEED = 42
    LENGTH = 64

    SAMPLES_PER_POINT = 20

    N_CRPS = [
        10 ** 5,
        10 ** 6,
        # 10 ** 7,
    ]

    STRUCTURES = [
        [2 ** 4] * 3,
        [2 ** 6] * 3,
        [2 ** 8] * 3,
    ]

    LEARNING_RATES = [
        0.0002,
        0.002,
        # 0.02,
    ]

    BATCH_FRAC = [0.05]

    def experiments(self):
        simulations = [
            simulation for i in range(self.SAMPLES_PER_POINT) for simulation in
            [
                Interpose3PUF(k_down=2, k_middle=1, k_up=1, n=self.LENGTH, seed=self.SEED + 1000 + i),
                Interpose3PUF(k_down=2, k_middle=2, k_up=2, n=self.LENGTH, seed=self.SEED + 2000 + i),
                Interpose3PUF(k_down=3, k_middle=1, k_up=1, n=self.LENGTH, seed=self.SEED + 3000 + i),
                Interpose3PUF(k_down=3, k_middle=3, k_up=3, n=self.LENGTH, seed=self.SEED + 4000 + i),
                Interpose3PUF(k_down=4, k_middle=1, k_up=1, n=self.LENGTH, seed=self.SEED + 5000 + i),
                Interpose3PUF(k_down=4, k_middle=4, k_up=4, n=self.LENGTH, seed=self.SEED + 6000 + i),
                # InterposeBinaryTree(k_down=1, k_middle=1, k_up=1, n=self.LENGTH, seed=self.SEED + 20000 + i),
                # InterposeCascade(ks=[1] * 8, n=self.LENGTH, seed=self.SEED + 40000 + i),
                # XORInterposePUF(k=4, n=self.LENGTH, seed=self.SEED + 60000 + i),
                # XORInterpose3PUF(k=3, n=self.LENGTH, seed=self.SEED + 80000 + i),
            ]
        ]
        return [
            ExperimentMLPScikitLearn(
                progress_log_prefix=self.name(),
                parameters=Parameters(
                    simulation=simulation,
                    seed=self.SEED + i,
                    N=N,
                    validation_frac=max(min(N // 20, self.MAX_NUM_VAL), self.MIN_NUM_VAL) / N,
                    preprocessing='short',
                    layers=layers,
                    learning_rate=learning_rate,
                    tolerance=0.0025,
                    patience=self.PATIENCE,
                    iteration_limit=self.ITERATION_LIMIT,
                    batch_size=int(N * batch_frac),
                )
            )
            for i, simulation in enumerate(simulations)
            for N in self.N_CRPS
            for layers in self.STRUCTURES
            for learning_rate in self.LEARNING_RATES
            for batch_frac in self.BATCH_FRAC
        ]

    def plot(self):
        data = self.experimenter.results
        data['Ncat'] = data.apply(lambda row: SplitAttackStudy._Ncat(row['N']), axis=1)
        data['size'] = data.apply(
            func=lambda row: '%s: k=%i' % (str(re.search(r'<([^\s]+)\s', str(row['simulation'])).group(1)),
                                           int(row['first_k'])),
            axis=1,
        )
        data['max_memory_gb'] = data.apply(lambda row: row['max_memory'] / 1024 ** 3, axis=1)
        data['Ne6'] = data.apply(lambda row: row['N'] / 1e6, axis=1)
        data = data.sort_values(['size', 'layers'])

        with axes_style('whitegrid'):
            params = dict(
                data=data,
                x='Ncat',
                y='accuracy',
                row='size',
                kind='swarm',
                aspect=2,
                height=4,
            )
            for name, params_ind in {
                'layer': dict(hue='layers', hue_order=[str([2 ** s] * 3) for s in range(2, 10)]),
                'learning_rate': dict(hue='learning_rate'),
            }.items():
                f = catplot(**params, **params_ind)
                f.axes.flatten()[0].set(ylim=(.45, 1.))
                f.savefig(f'figures/{self.name()}.{name}.pdf')
                close(f.fig)
