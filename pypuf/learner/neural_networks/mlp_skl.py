"""
This module provides a learning algorithm for Arbiter PUFs using a Multilayer Perceptron and Adam from Scikit-learn.
This approach is based on the work of Aseeri et. al. "A Machine Learning-based Security Vulnerability Study on XOR PUFs
for Resource-Constraint Internet of Things", 2018 IEEE International Congress on Internet of Things (ICIOT),
San Francisco, CA, 2018, pp. 49-56.
"""

from numpy import reshape, mean, abs, sign
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from numpy.random import RandomState
from pypuf.learner.base import Learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import ChallengeResponseSet


class MultiLayerPerceptronScikitLearn(Learner):
    """
    Define Learner that generates a Multilayer Perceptron model which uses the Adam optimization method.
    """

    SEED_RANGE = 2 ** 32

    def __init__(self, n, k, training_set, validation_frac, transformation, preprocessing, layers=(10, 10),
                 activation='relu', domain_in=-1, learning_rate=0.001, penalty=0.0002, beta_1=0.9, beta_2=0.999,
                 tolerance=0.0025, patience=4, print_learning=False, iteration_limit=40, batch_size=1000,
                 seed_model=0xc0ffee):
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

    def prepare(self):
        """
        Preprocess training data and initialize the Multilayer Perceptron.
        """
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
        if self.domain_in == 0:
            self.training_set.challenges = (self.training_set.challenges + 1) / 2
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

        class Model:
            """
            This defines a wrapper for the learning model to fit in with the Learner framework.
            """
            def __init__(self, nn, n, k, preprocess, domain_in):
                self.nn = nn
                self.n = n
                self.k = k
                self.preprocess = preprocess
                self.domain_in = domain_in

            def eval(self, cs):
                cs_preprocessed = self.preprocess(challenges=cs, k=self.k)
                challenges = cs_preprocessed if self.domain_in == -1 else (cs_preprocessed + 1) / 2
                predictions = self.nn.predict(X=challenges)
                predictions_1_1 = predictions * 2 - 1
                return sign(predictions_1_1).flatten()

        self.model = Model(
            nn=self.nn,
            n=self.n,
            k=self.k,
            preprocess=preprocess,
            domain_in=self.domain_in
        )

    def learn(self):
        """
        Train the model with early stopping.
        """
        def accuracy(y_true, y_pred):
            return (1 + mean(y_true * y_pred)) / 2
        x, x_val, y, y_val = train_test_split(
            self.training_set.challenges,
            self.training_set.responses,
            random_state=self.seed_model,
            test_size=self.validation_frac,
            stratify=self.training_set.responses,
        )
        counter = 0
        threshold = 0
        best = 0
        waiting = min(8, max(4, self.k))
        for epoch in range(self.iteration_limit):
            self.nn = self.nn.partial_fit(
                X=self.training_set.challenges,
                y=self.training_set.responses,
                classes=[-1, 1],
            )
            tmp = accuracy(y_true=y_val, y_pred=self.model.eval(cs=x_val))
            self.accuracy_curve.append(tmp)
            if epoch < waiting:
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
