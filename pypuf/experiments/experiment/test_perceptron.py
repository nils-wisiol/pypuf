"""
    This module provides a primer test for the Perceptron Learner.
    It generates training and test sets from PUF simulation and
    trains/tests a Perceptron learner using monomials to transform the
    feature space.
"""
from typing import NamedTuple
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.perceptron.perceptron import Perceptron
from pypuf import tools

from uuid import UUID


class Parameters(NamedTuple):
    """
        Paramters that are being passed to the Experiment constructor.
    """
    n: int  # Number of challenge bits
    k: int  # Number of XOR arbiters
    N: int  # Number of samples generated for training/testing
    batch_size: int
    epochs: int


class Result(NamedTuple):
    """
        Results that are returned and printed after testing.
    """
    experiment_id: UUID
    accuracy: float


class TestPerceptron(Experiment):
    """
        Testing the Perceptron Learner with k=1 Arbiter PUF
    """

    def __init__(self, progress_log_prefix, parameters):

        progress_log_name = None if not progress_log_prefix else 'PERCEPTRON_k{}_n{}'.format(parameters.k, parameters.n)

        super().__init__(progress_log_name, parameters)
        self.train_set = None
        self.valid_set = None
        self.simulation = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: Initialize Perceptron, split training sets, etc.
        """
        # Create simulation that generates PUF challenge-response pairs
        self.simulation = LTFArray(
            weight_array=LTFArray.normal_weights(
                self.parameters.n,
                self.parameters.k
                ),
            transform='atf',  # todo: make dynamic
            combiner='xor'
        )
        # Generate training and test sets from PUF simulation
        N_valid = max(min(self.parameters.N // 20, 10000), 200)  # ???@chris
        N_train = self.parameters.N - N_valid
        self.train_set = tools.TrainingSet(self.simulation, N_train)
        self.valid_set = tools.TrainingSet(self.simulation, N_valid)

        # H4rdcode monomials : todo - outsource and make dep. on n, k
        assert self.parameters.k == 1
        n = self.parameters.n
        monomials = [range(i, n) for i in range(n)]

        self.learner = Perceptron(self.train_set, self.valid_set,
                                  monomials=monomials,
                                  batch_size=self.parameters.batch_size,
                                  epochs=self.parameters.epochs)

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
