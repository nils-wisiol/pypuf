"""
This module provides a learning algorithm for Arbiter PUFs using a Multilayer Perceptron and Adam from Scikit-learn.
This approach is based on the work of Aseeri et. al. "A Machine Learning-based Security Vulnerability Study on XOR PUFs
for Resource-Constraint Internet of Things", 2018 IEEE International Congress on Internet of Things (ICIOT),
San Francisco, CA, 2018, pp. 49-56.
"""
from numpy import reshape, mean, sign, shape
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from pypuf.learner.base import Learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.simulation.base import Simulation


class MultiLayerPerceptronScikitLearn(Learner):
    """
    Define Learner that generates a Multilayer Perceptron model which uses the Adam optimization method.
    """

    SEED_RANGE = 2 ** 32
    DELAY = 8

    def __init__(self, n, k, training_set, validation_frac, transformation, preprocessing, layers=(10, 10),
                 activation='relu', domain_in=-1, learning_rate=0.001, penalty=0.0002, beta_1=0.9, beta_2=0.999,
                 tolerance=0.0025, patience=4, print_learning=False, iteration_limit=40, batch_size=1000,
                 seed_model=0xc0ffee, logger=None):
        """
        :param n:               int; positive, length of PUF (typically power of 2)
        :param k:               int; positive, width of PUF (typically between 1 and 8)
        :param training_set:    object holding two arrays (challenges, responses); elements are -1 and 1
        :param validation_frac: float: ratio of validation set size to training_set size
        :param transformation:  function; changes the challenges (while expanding them to the size of k * n)
        :param preprocessing:   string: 'no', 'short', 'full'; determines how the learner changes the challenges
        :param layers:          list of ints; number of nodes within hidden layers
        :param activation:      function or string;
        :param domain_in:       int: -1, 0; determines if the challenges are in {-1, 1} or {0, 1} domain
        :param learning_rate:   float; between 0 and 1, parameter of Adam optimizer
        :param penalty:         float; between 0 and 1, parameter of l2 regularization term used in learning
        :param beta_1:          float; between 0 and 1, parameter of Adam optimizer
        :param beta_2:          float; between 0 and 1, parameter of Adam optimizer
        :param tolerance:       float; between 0 and 1, stopping criterium: minimum change of loss
        :param patience:        int; positive, stopping criterion: maximum number of epochs with low change of loss
        :param iteration_limit: int; positive, maximum number of epochs
        :param batch_size:      int; between 1 and N, number of training samples used until updating the model's weights
        :param seed_model:      int; between 1 and 2**32, seed for PRNG
        :param print_learning:  bool; whether to print learning progress of tensorflow
        """
        self.n = n
        self.k = k
        self.training_set = training_set
        self.validation_frac = validation_frac
        self.transformation = transformation
        self.preprocessing = preprocessing
        self.layers = layers
        self.activation = activation
        self.domain_in = domain_in
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
        self.accuracy_curve = []
        self.nn = None
        self.model = None
        self.log = logger

    def prepare(self):
        """
        Initialize the Multilayer Perceptron.
        """
        self.nn = MLPClassifier(
            solver='adam',
            alpha=self.penalty,
            hidden_layer_sizes=self.layers,
            random_state=self.seed_model,
            learning_rate='constant',
            learning_rate_init=self.learning_rate,
            batch_size=self.batch_size,
            shuffle=True,  # a bug in scikit learn stops us from shuffling, no matter what we set here
            activation=self.activation,
            verbose=self.print_learning,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )

        class Model(Simulation):
            """
            This defines a wrapper for the learning model to fit in with the Learner framework.
            """

            def __init__(self, nn, n, k, preprocess, domain_in):
                self.nn = nn
                self.n = n
                self.k = k
                self.preprocess = preprocess
                self.domain_in = domain_in

            def challenge_length(self) -> int:
                return self.n

            def response_length(self) -> int:
                return 1

            def eval(self, cs):
                cs = self.preprocess(challenges=cs, k=self.k)
                cs = cs if self.domain_in == -1 else (cs + 1) / 2
                cs_shape = shape(cs)
                if len(cs_shape) == 3:
                    cs = reshape(cs, (cs_shape[0], cs_shape[1] * cs_shape[2]))
                return sign(self.nn.predict(X=cs) * 2 - 1).flatten()

        self.model = Model(
            nn=self.nn,
            n=self.n,
            k=self.k,
            preprocess=LTFArray.preprocess(transformation=self.transformation, kind=self.preprocessing),
            domain_in=self.domain_in
        )

    def learn(self):
        """
        Train the model with early stopping.
        """
        self.log(f'learner tarted, splitting training set')
        x, x_val, y, y_val = train_test_split(
            self.training_set.challenges,
            self.training_set.responses,
            random_state=self.seed_model,
            test_size=self.validation_frac,
            stratify=self.training_set.responses,
        )
        preprocess = LTFArray.preprocess(transformation=self.transformation, kind=self.preprocessing)
        if self.preprocessing != 'no':
            self.log(f'preprocessing training set')
            x = preprocess(challenges=x, k=self.k)
            if self.preprocessing == 'full':
                x = reshape(x, (len(x), self.k * self.n))
        if self.domain_in == 0:
            x = (x + 1) / 2

        def accuracy(y_true, y_pred):
            return (1 + mean(y_true * y_pred)) / 2
        counter = 0
        threshold = 0
        best = 0
        self.log(f'starting learning loop with at most {self.iteration_limit} iterations')
        for epoch in range(self.iteration_limit):
            self.nn = self.nn.partial_fit(
                X=x,
                y=y,
                classes=[-1, 1],
            )
            tmp = accuracy(y_true=y_val, y_pred=self.model.eval(cs=x_val))
            self.accuracy_curve.append(tmp)
            self.log(f'epoch {epoch} accuracy {tmp}')
            if epoch < self.DELAY:
                continue
            if tmp > best:
                best = tmp
                if tmp >= threshold + self.tolerance:
                    threshold = best
                    counter = 0
                    continue
            counter += 1
            if counter >= self.patience:
                break
        return self.model
