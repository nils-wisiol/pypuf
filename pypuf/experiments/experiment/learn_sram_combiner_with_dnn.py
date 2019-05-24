"""
    This module tries to learn a k-Arbiter PUF with SRAM combiner.
    It generates training and test sets from PUF simulation and
    trains/tests a DNN learner.
"""
from typing import NamedTuple
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.neural_network.deep_neural_network import DeepNeuralNetwork
from pypuf import tools

from numpy.random.mtrand import RandomState
from numpy import sign

from uuid import UUID


class Parameters(NamedTuple):
    """
        Paramters that are being passed to the Experiment constructor.
    """
    n: int  # Number of challenge bits
    k: int  # Number of arbiters -> SRAM size = 2**k
    N: int  # Number of samples generated for training/testing
    batch_size: int
    epochs: int


class Result(NamedTuple):
    """
        Results that are returned and printed after testing.
    """
    experiment_id: UUID
    accuracy: float


class SRAMDNN(Experiment):
    """
        Learn the k-Arbiter PUF with SRAM Combiner using DNN.
    """

    def __init__(self, progress_log_prefix, parameters):

        progress_log_name = None if not progress_log_prefix else 'SRAMDNN_k{}_n{}'.format(parameters.k, parameters.n)

        super().__init__(progress_log_name, parameters)
        self.train_set = None
        self.valid_set = None
        self.simulation = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: Initialize DNN and PUFs, split training sets, etc.
        """
        # Create simulation that generates PUF challenge-response pairs
        k = self.parameters.k
        sram = RandomState(seed=0xdeadbeef).normal(size=2**k)
        sram = sign(sram)
        self.simulation = LTFArray(
            weight_array=LTFArray.normal_weights(
                self.parameters.n,
                self.parameters.k
                ),
            transform='atf',
            combiner='xor'
            #lut=sram
        )
        # Generate training and test sets from PUF simulation
        N_valid = max(min(self.parameters.N // 20, 10000), 200)  # ???@chris
        N_train = self.parameters.N - N_valid
        self.train_set = tools.TrainingSet(self.simulation, N_train)
        self.valid_set = tools.TrainingSet(self.simulation, N_valid)

        self.learner = DeepNeuralNetwork(self.train_set, self.valid_set,
                                  monomials=None,
                                  batch_size=self.parameters.batch_size,
                                  epochs=self.parameters.epochs,
                                  parameters=self.parameters)

    def run(self):
        """
        Start the training of the Perceptron
        """
        self.prepare()
        self.model = self.learner.learn()

    def analyze(self):
        assert self.model is not None
        accuracy = self.learner.history.history['val_pypuf_accuracy'][-1]
        return Result(experiment_id=self.id,
                      accuracy=accuracy)
