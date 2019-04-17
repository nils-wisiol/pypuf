from typing import NamedTuple, List
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.perceptron import Perceptron
from pypuf import tools


class Parameters(NamesTuple):
    n: int # Number of challenge bits
    k: int # Number of XOR arbiters
    N: int # Number of samples generated for training/testing


class TestPerceptron(Experiment):
    """
    Testing the Perceptron Learner with k=1 Arbiter PUF
    """

    def __init__(self, progress_log_prefix, parameters):

        progress_log_name = None if not progress_log_prefix else 'PERCEPTRON_k{}_n{}'.format(paramters.k, parameters.n)

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
            transform='atf', # todo: make dynamic
            combiner='xor'
        )
        # Generate training and test sets from PUF simulation
        N_valid = max(min(self.parameters.N // 20, 10000), 200) # ???@chris
        N_train = self.parameters.N - N_valid
        self.train_set = tools.TrainingSet(self.simulation, N_train)
        self.valid_set = tools.TrainingSet(self.simulation, N_valid)

        # H4rdcode monomials : todo - outsource and make dep. on n, k
        assert(self.parameters.k == 1)
        n = self.parameters.n
        monomials = [range(i:n) for i in range(n)]

        self.learner = Perceptron(self.train_set, self.valid_set,
                                  monomials=monomials)

    def run(self):
        """
        Start the training of the Perceptron
        """
        self.prepare()
        self.model = self.learner.learn()

    def analyze(self):
        assert self.model is not None
        accuracy = 1.0 - tools.approx_dist(
                self.simulation,
                self.model,
                min(10000, 2 ** self.parameters.n)
        )
        return accuracy

