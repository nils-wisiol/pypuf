from sklearn.neural_network import MLPClassifier
from numpy.random import RandomState
from pypuf.learner.base import Learner
from pypuf.tools import random_inputs


class MLP(Learner):

    def __init__(self, N, n, k, simulation, iteration_limit,
                 seed_distance=0xc0ffee, seed_model=0xbeef,
                 batch_size=1000, learning_rate=1e-3, activation='relu',
                 hidden_layer_sizes=(10, 10), tol=0.01, n_iter_no_change=1):
        self.N = N
        self.n = n
        self.k = k
        self.simulation = simulation
        self.iteration_limit = iteration_limit
        self.seed_distance = seed_distance
        self.seed_model = seed_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.challenges = None
        self.responses = None
        self.clf = None
        self.model = None

    def prepare(self):
        self.challenges = random_inputs(self.n, self.N, RandomState(seed=self.seed_distance))
        self.responses = self.simulation.eval(self.challenges)
        self.clf = MLPClassifier(
            solver='adam',
            hidden_layer_sizes=self.hidden_layer_sizes,
            random_state=self.seed_model,
            learning_rate_init=self.learning_rate,
            batch_size=self.batch_size,
            shuffle=False,  # a bug in scikit learn stops us from shuffling, no matter what we set here
            activation=self.activation,
            max_iter=self.iteration_limit,
            tol=self.tol,
            n_iter_no_change=self.n_iter_no_change,
        )

        class MLPResult:
            def __init__(self, n, clf):
                self.n = n
                self.clf = clf

            def eval(self, cs):
                return self.clf.predict(cs)

        self.model = MLPResult(self.n, self.clf)

    def learn(self):
        self.clf.fit(self.challenges, self.responses)
        return self.model
