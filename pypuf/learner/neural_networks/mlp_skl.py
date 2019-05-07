from numpy import reshape
from sklearn.neural_network import MLPClassifier
from numpy.random import RandomState
from pypuf.learner.base import Learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import ChallengeResponseSet


class MultiLayerPerceptronScikitLearn(Learner):

    SEED_RANGE = 2 ** 32

    def __init__(self, n, k, training_set, validation_frac, transformation, preprocessing, layers=(10, 10),
                 activation='relu', learning_rate=0.001, penalty=0.0001, beta_1=0.9, beta_2=0.999, tolerance=0.001,
                 patience=5, print_learning=False, iteration_limit=100, batch_size=1000, seed_model=0xc0ffee):
        self.n = n
        self.k = k
        self.training_set = training_set
        self.validation_frac = validation_frac
        self.transformation = transformation
        self.preprocessing = preprocessing
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
        self.print_learning = print_learning
        self.nn = None
        self.model = None

    def prepare(self):
        in_shape = self.n
        preprocess = LTFArray.preprocess(transformation=self.transformation, kind=self.preprocessing)
        if self.preprocessing != 'no':
            self.training_set = ChallengeResponseSet(
                challenges=preprocess(challenges=self.training_set.challenges, k=self.k),
                responses=self.training_set.responses
            )
            if self.preprocessing == 'full':
                in_shape = self.k * self.n
            self.training_set.challenges = reshape(self.training_set.challenges, (self.training_set.N, in_shape))
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
            verbose=self.print_learning,
            early_stopping=True,
            validation_fraction=self.validation_frac,
            n_iter_no_change=self.patience,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )

        class Model:
            def __init__(self, n, nn, preprocess):
                self.n = n
                self.nn = nn
                self.preprocess = preprocess

            def eval(self, cs):
                return self.nn.predict(X=cs)

        self.model = Model(n=self.n, nn=self.nn, preprocess=preprocess)

    def learn(self):
        self.nn.fit(X=self.training_set.challenges, y=self.training_set.responses)
        return self.model
