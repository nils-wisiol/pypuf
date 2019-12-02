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

    def __init__(self, n: int, k_up: int, k_middle: int, k_down: int, seed: int, noisiness: float = 0) -> None:
        self.seed = seed
        self.prng = RandomState(seed)
        self.n = n
        self.k = k_up
        self.k_up, self.k_middle, self.k_down = k_up, k_middle, k_down
        self.noisiness = noisiness
        seeds = [self.prng.randint(0, 2 ** 32) for _ in range(6)]
        self.up = XORArbiterPUF(n=n, k=k_up, seed=seeds[0], noisiness=noisiness, noise_seed=seeds[1])
        self.middle = XORArbiterPUF(n=n + 1, k=k_up, seed=seeds[2], noisiness=noisiness, noise_seed=seeds[3])
        self.down = XORArbiterPUF(n=n + 1, k=k_up, seed=seeds[4], noisiness=noisiness, noise_seed=seeds[5])
        self.interpose_pos = n // 2

    def __repr__(self) -> str:
        return f'Interpose3PUF, n={self.n}, k_up={self.k_up}, k_middle={self.k_middle}, k_down={self.k_down}, ' \
               f'pos={self.interpose_pos}'

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

    def __init__(self, n: int, ks: List[int], seed: int, noisiness: float = 0) -> None:
        self.seed = seed
        self.prng = RandomState(seed)
        self.n = n
        self.ks = ks
        self.k = ks[0]
        self.depth = len(ks)
        self.noisiness = noisiness
        self.layers = [[
            XORArbiterPUF(
                n=n + 1 if i > 0 else n,
                k=ks[i],
                seed=self.prng.randint(0, 2 ** 32),
                noisiness=noisiness,
                noise_seed=self.prng.randint(0, 2 ** 32)
            )
            for _ in range(2 ** i)
        ]
            for i in range(self.depth)
        ]
        self.interpose_pos = n // 2

    def __repr__(self) -> str:
        return f'InterposeBinaryTree, n={self.n}, k={self.k}, depth={self.depth}, pos={self.interpose_pos}'

    def challenge_length(self) -> int:
        return self.layers[0][0].challenge_length()

    def response_length(self) -> int:
        return 1

    def _interpose(self, challenges, bits):
        pos = self.interpose_pos
        return concatenate(
            (challenges[:, :pos], bits.reshape(-1, 1), challenges[:, pos:]),
            axis=1,
        )

    def eval(self, challenges: ndarray) -> ndarray:
        responses = [self.layers[0][0].eval(challenges=challenges)]
        for i in range(self.depth - 1):
            responses = [self.layers[i + 1][j].eval(
                challenges=self._interpose(challenges=challenges, bits=responses[int(j / 2)])
            ) for j in range(len(self.layers[i + 1]))]
        return prod(responses, axis=0)


class InterposeCascade(Simulation):

    def __init__(self, n: int, ks: List[int], seed: int, noisiness: float = 0) -> None:
        self.seed = seed
        self.prng = RandomState(seed)
        self.n = n
        self.k = ks[0]
        self.ks = ks
        self.noisiness = noisiness
        seeds = [self.prng.randint(0, 2 ** 32) for _ in range(2 * len(ks))]
        self.layers = [
            XORArbiterPUF(
                n=n + 1 if i > 0 else n,
                k=k,
                seed=seeds[2 * i],
                noisiness=noisiness,
                noise_seed=seeds[2 * i + 1],
            )
            for i, k in enumerate(ks)
        ]
        self.interpose_pos = n // 2

    def __repr__(self) -> str:
        return f'InterposeCascade, n={self.n}, ks={str(self.ks)}, pos={self.interpose_pos}'

    def challenge_length(self) -> int:
        return self.layers[0].challenge_length()

    def response_length(self) -> int:
        return 1

    def _interpose(self, challenges, bits):
        pos = self.interpose_pos
        return concatenate(
            (challenges[:, :pos], bits.reshape(-1, 1), challenges[:, pos:]),
            axis=1,
        )

    def eval(self, challenges: ndarray) -> ndarray:
        result = 1
        for i, layer in enumerate(self.layers):
            result = result * layer.eval(self._interpose(challenges=challenges, bits=result) if i > 0 else challenges)
        return result


class XORInterposePUF(Simulation):

    def __init__(self, n: int, k: int, seed: int, noisiness: float = 0) -> None:
        self.seed = seed
        self.prng = RandomState(seed)
        self.n = n
        self.k = k
        self.noisiness = noisiness
        seeds = [self.prng.randint(0, 2 ** 32) for _ in range(4 * k)]
        self.layers_up = [
            XORArbiterPUF(n=n, k=1, seed=seeds[2 * i], noisiness=noisiness, noise_seed=seeds[2 * i + 1])
            for i in range(k)
        ]
        self.layers_down = [
            XORArbiterPUF(n=n + 1, k=1, seed=seeds[2 * (i + k)], noisiness=noisiness, noise_seed=seeds[2 * (i + k) + 1])
            for i in range(k)
        ]
        self.interpose_pos = n // 2

    def __repr__(self) -> str:
        return f'XORInterposePUF, n={self.n}, k={self.k}, pos={self.interpose_pos}'

    def challenge_length(self) -> int:
        return self.layers_up[0].challenge_length()

    def response_length(self) -> int:
        return 1

    def _interpose(self, challenges, bits):
        pos = self.interpose_pos
        return concatenate(
            (challenges[:, :pos], bits.reshape(-1, 1), challenges[:, pos:]),
            axis=1,
        )

    def eval(self, challenges: ndarray) -> ndarray:
        return prod(
            a=[self.layers_down[i].eval(self._interpose(
                challenges=challenges,
                bits=self.layers_up[i].eval(challenges)
            )) for i in range(self.k)],
            axis=0,
        )


class XORInterpose3PUF(Simulation):

    def __init__(self, n: int, k: int, seed: int, noisiness: float = 0) -> None:
        self.seed = seed
        self.prng = RandomState(seed)
        self.n = n
        self.k = k
        self.noisiness = noisiness
        seeds = [self.prng.randint(0, 2 ** 32) for _ in range(6 * k)]
        self.layers_up = [
            XORArbiterPUF(n=n, k=1, seed=seeds[2 * i], noisiness=noisiness, noise_seed=seeds[2 * i + 1])
            for i in range(k)
        ]
        self.layers_middle = [
            XORArbiterPUF(n=n + 1, k=1, seed=seeds[2 * (i + k)], noisiness=noisiness, noise_seed=seeds[2 * (i + k) + 1])
            for i in range(k)
        ]
        self.layers_down = [
            XORArbiterPUF(n=n + 1, k=1, seed=seeds[2 * (i+2*k)], noisiness=noisiness, noise_seed=seeds[2 * (i+2*k) + 1])
            for i in range(k)
        ]
        self.interpose_pos = n // 2

    def __repr__(self) -> str:
        return f'XORInterpose3PUF, n={self.n}, k={self.k}, pos={self.interpose_pos}'

    def challenge_length(self) -> int:
        return self.layers_up[0].challenge_length()

    def response_length(self) -> int:
        return 1

    def _interpose(self, challenges, bits):
        pos = self.interpose_pos
        return concatenate(
            (challenges[:, :pos], bits.reshape(-1, 1), challenges[:, pos:]),
            axis=1,
        )

    def eval(self, challenges: ndarray) -> ndarray:
        return prod(
            a=[self.layers_down[i].eval(self._interpose(
                challenges=challenges,
                bits=self.layers_middle[i].eval(self._interpose(
                    challenges=challenges,
                    bits=self.layers_up[i].eval(challenges)
                ))
            )) for i in range(self.k)],
            axis=0,
        )


class Parameters(NamedTuple):
    """
    Define all input parameters for the Experiment.
    """
    simulation: Simulation
    seed_simulation: int
    noisiness: float
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
    stability: float
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
        self.stability = 1.0
        self.training_set = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.stability = 1.0 - tools.approx_dist(
            instance1=self.simulation,
            instance2=self.simulation,
            num=10 ** 4,
            random_instance=RandomState(seed=self.parameters.seed),
        )
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
        if self.stability < 0.65:
            self.progress_logger.debug(f'The stability of the target is too low: {self.stability}')
            return
        self.progress_logger.debug('Starting learner')
        self.model = self.learner.learn()

    def analyze(self):
        """
        Calculate statistics of the trained model and write them into a result log file.
        """
        self.progress_logger.debug('Analyzing result')
        accuracy = -1 if not self.model else 1.0 - tools.approx_dist(
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
            iterations=-1 if not self.model else self.learner.nn.n_iter_,
            accuracy=accuracy,
            stability=self.stability,
            loss_curve=[-1] if not self.model else [round(loss, 3) for loss in self.learner.nn.loss_curve_],
            accuracy_curve=[-1] if not self.model else [round(accuracy, 3) for accuracy in self.learner.accuracy_curve],
            max_memory=self.max_memory(),
        )


class InterposeMLPStudy(Study):
    """
    Define a set of experiments by combining several sets of hyperparameters.
    """
    SHUFFLE = True

    ITERATION_LIMIT = 400
    PATIENCE = ITERATION_LIMIT
    MAX_NUM_VAL = 10000
    MIN_NUM_VAL = 200
    PRINT_LEARNING = False
    LENGTH = 64
    SEED = 42
    NOISINESS = 0.1

    SAMPLES_PER_POINT = 10

    N_CRPS = {
        'small': [
            10 * 1000,
            50 * 1000,
            200 * 1000,
        ],
        'medium': [
            50 * 1000,
            200 * 1000,
            1000 * 1000,
        ],
        'large': [
            200 * 1000,
            1000 * 1000,
            10000 * 1000,
        ],
    }

    STRUCTURES = {
        'small': [
            [2 ** 3] * 3,
            [2 ** 4] * 3,
            [2 ** 3] * 4,
        ],
        'medium': [
            [2 ** 5] * 3,
            [2 ** 6] * 3,
            [2 ** 5] * 4,
        ],
        'large': [
            [2 ** 7] * 3,
            [2 ** 8] * 3,
            [2 ** 7] * 4,
        ],
    }

    LEARNING_RATES = [
        0.0002,
        0.002,
        0.02,
    ]

    BATCH_FRAC = [0.05]

    def experiments(self):
        simulations = {
            'small': [
                simulation for i in range(self.SAMPLES_PER_POINT) for simulation in
                [
                    Interpose3PUF(self.LENGTH, 2, 1, 1, (self.SEED + 1000 + i) % 2 ** 32, self.NOISINESS),
                    Interpose3PUF(self.LENGTH, 2, 2, 2, (self.SEED + 2000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeBinaryTree(self.LENGTH, [1, 1, 1], (self.SEED + 20000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeBinaryTree(self.LENGTH, [1, 1, 2], (self.SEED + 21000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [1] * 2, (self.SEED + 40000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [2] * 2, (self.SEED + 41000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [1] * 3, (self.SEED + 42000 + i) % 2 ** 32, self.NOISINESS),
                    XORInterposePUF(self.LENGTH, 2, (self.SEED + 60000 + i) % 2 ** 32, self.NOISINESS),
                    XORInterpose3PUF(self.LENGTH, 2, (self.SEED + 80000 + i) % 2 ** 32, self.NOISINESS),
                ]
            ],
            'medium': [
                simulation for i in range(self.SAMPLES_PER_POINT) for simulation in
                [
                    Interpose3PUF(self.LENGTH, 3, 1, 1, (self.SEED + 3000 + i) % 2 ** 32, self.NOISINESS),
                    Interpose3PUF(self.LENGTH, 3, 3, 3, (self.SEED + 4000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [2] * 3, (self.SEED + 43000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [1] * 4, (self.SEED + 44000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [2] * 4, (self.SEED + 45000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [1] * 5, (self.SEED + 46000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [1] * 6, (self.SEED + 47000 + i) % 2 ** 32, self.NOISINESS),
                    XORInterposePUF(self.LENGTH, 3, (self.SEED + 61000 + i) % 2 ** 32, self.NOISINESS),
                    XORInterpose3PUF(self.LENGTH, 3, (self.SEED + 81000 + i) % 2 ** 32, self.NOISINESS),
                ]
            ],
            'large': [
                simulation for i in range(self.SAMPLES_PER_POINT) for simulation in
                [
                    Interpose3PUF(self.LENGTH, 4, 1, 1, (self.SEED + 5000 + i) % 2 ** 32, self.NOISINESS),
                    Interpose3PUF(self.LENGTH, 4, 4, 4, (self.SEED + 6000 + i) % 2 ** 32, self.NOISINESS),
                    Interpose3PUF(self.LENGTH, 5, 1, 1, (self.SEED + 7000 + i) % 2 ** 32, self.NOISINESS),
                    Interpose3PUF(self.LENGTH, 5, 5, 5, (self.SEED + 8000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeBinaryTree(self.LENGTH, [2, 2, 2], (self.SEED + 22000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeBinaryTree(self.LENGTH, [1, 1, 1, 1], (self.SEED + 22000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [2] * 5, (self.SEED + 48000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [3] * 3, (self.SEED + 49000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [4] * 2, (self.SEED + 50000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [2] * 5, (self.SEED + 51000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [1] * 7, (self.SEED + 52000 + i) % 2 ** 32, self.NOISINESS),
                    InterposeCascade(self.LENGTH, [1] * 8, (self.SEED + 53000 + i) % 2 ** 32, self.NOISINESS),
                    XORInterposePUF(self.LENGTH, 4, (self.SEED + 62000 + i) % 2 ** 32, self.NOISINESS),
                    XORInterpose3PUF(self.LENGTH, 4, (self.SEED + 82000 + i) % 2 ** 32, self.NOISINESS),
                ]
            ],
        }
        return [
            ExperimentMLPScikitLearn(
                progress_log_prefix=self.name(),
                parameters=Parameters(
                    simulation=simulation,
                    seed_simulation=simulation.seed,
                    noisiness=simulation.noisiness,
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
            for group in self.N_CRPS.keys()
            for i, simulation in enumerate(simulations[group])
            for N in self.N_CRPS[group]
            for layers in self.STRUCTURES[group]
            for learning_rate in self.LEARNING_RATES
            for batch_frac in self.BATCH_FRAC
        ]

    def plot(self):
        data = self.experimenter.results
        data['Ncat'] = data.apply(lambda row: SplitAttackStudy._Ncat(row['N']), axis=1)
        data['max_memory_gb'] = data.apply(lambda row: row['max_memory'] / 1024 ** 3, axis=1)
        data['Ne6'] = data.apply(lambda row: row['N'] / 1e6, axis=1)

        with axes_style('whitegrid'):
            params = dict(
                data=data,
                x='Ncat',
                y='accuracy',
                row='simulation',
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
