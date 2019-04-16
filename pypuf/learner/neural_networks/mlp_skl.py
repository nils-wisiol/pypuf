from sklearn.neural_network import MLPClassifier
from numpy.random import RandomState
from pypuf.learner.base import Learner


class MultiLayerPerceptronScikitLearn(Learner):

    SEED_RANGE = 2 ** 32

    def __init__(self, n, k, training_set, validation_frac, transformation=None, layers=(10, 10), activation='relu',
                 learning_rate=0.001, penalty=0.0001, beta_1=0.9, beta_2=0.999, tolerance=0.001, patience=5,
                 iteration_limit=100, batch_size=1000, seed_model=0xc0ffee):
        self.n = n
        self.k = k
        self.training_set = training_set
        self.validation_frac = validation_frac
        self.transformation = transformation
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tolerance = tolerance
        self.patience = patience
        self.iteration_limit = iteration_limit
        self.batch_size = min(batch_size, training_set.N)
        self.seed_model = RandomState(seed_model).randint(self.SEED_RANGE)
        self.nn = None
        self.model = None

    def prepare(self):
        self.nn = MLPClassifier(
            solver='adam',
            alpha=self.penalty,
            hidden_layer_sizes=self.layers,
            random_state=self.seed_model,
            learning_rate_init=self.learning_rate,
            batch_size=self.batch_size,
            shuffle=False,  # a bug in scikit learn stops us from shuffling, no matter what we set here
            activation=self.activation,
            max_iter=self.iteration_limit,
            tol=self.tolerance,
            early_stopping=True,
            validation_fraction=self.validation_frac,
            n_iter_no_change=self.patience,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )

        class Model:
            def __init__(self, n, clf):
                self.n = n
                self.clf = clf

            def eval(self, cs):
                return self.clf.predict(cs)

        self.model = Model(self.n, self.nn)

    def learn(self):
        self.nn.fit(self.training_set.challenges, self.training_set.responses)
        return self.model
