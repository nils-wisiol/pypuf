"""
This module provides a learning algorithm for Arbiter PUFs using a Multilayer Perceptron and Adam from Tensorflow.
This approach is based on the work of Aseeri et. al. "A Machine Learning-based Security Vulnerability Study on XOR PUFs
for Resource-Constraint Internet of Things", 2018 IEEE International Congress on Internet of Things (ICIOT),
San Francisco, CA, 2018, pp. 49-56.
"""

from os import environ

from numpy import reshape, sign
from numpy.random import seed
from numpy.random.mtrand import RandomState
from tensorflow import set_random_seed, ConfigProto, get_default_graph, Session, multiply
from tensorflow.python.platform.tf_logging import set_verbosity
from tensorflow.python.training.tensorboard_logging import ERROR
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.backend import maximum as tf_max, mean as tf_mean, abs as tf_abs, set_session, log
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

from pypuf.learner.base import Learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import ChallengeResponseSet


environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_verbosity(ERROR)


class MultiLayerPerceptronTensorflow(Learner):
    """
    Define Learner that generates a Multilayer Perceptron model which uses the Adam optimization method.
    """

    SEED_RANGE = 2 ** 32
    EPSILON = 1e-6  # should not be smaller, else NaN in log

    def __init__(self, n, k, training_set, validation_set, transformation, preprocessing, layers=(10, 10),
                 activation='relu', loss='squared_hinge', domain_in=-1, domain_out=-1, learning_rate=0.001,
                 penalty=0.0002, beta_1=0.9, beta_2=0.999, tolerance=0.0025, patience=4, print_learning=False,
                 iteration_limit=40, batch_size=1000, seed_model=0xc0ffee):
        """
        :param n:               int; positive, length of PUF (typically power of 2)
        :param k:               int; positive, width of PUF (typically between 1 and 8)
        :param training_set:    object holding two arrays (challenges, responses); elements are -1 and 1
        :param validation_set:  same as training_set, but only used for validation (not for training)
        :param transformation:  function; changes the challenges (while expanding them to the size of k * n)
        :param preprocessing:   string: 'no', 'short', 'full'; determines how the learner changes the challenges
        :param layers:          list of ints; number of nodes within hidden layers
        :param activation:      function or string;
        :param loss:            string: 'squared_hinge', 'log_loss'; loss function used in learning
        :param domain_in:       int: -1, 0; determines if the challenges are in {-1, 1} or {0, 1} domain
        :param domain_out:      int: -1, 0; determines if the responses are in {-1, 1} or {0, 1} domain
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
        self.validation_set = validation_set
        self.transformation = transformation
        self.preprocessing = preprocessing
        self.layers = layers
        self.activation = activation
        self.loss = loss
        self.domain_in = domain_in
        self.domain_out = domain_out
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tolerance = tolerance
        self.patience = patience
        self.iteration_limit = iteration_limit
        self.batch_size = min(batch_size, training_set.N)
        self.seed_model = RandomState(seed_model).randint(self.SEED_RANGE)
        self.print_learning = 0 if not print_learning else 1
        self.nn = None
        self.history = None
        self.model = None

    def prepare(self):
        """
        Preprocess training data and initialize the Multilayer Perceptron.
        """
        seed(self.seed_model)
        set_random_seed(self.seed_model)
        session_conf = ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        set_session(Session(graph=get_default_graph(), config=session_conf))
        in_shape = self.n
        preprocess = LTFArray.preprocess(transformation=self.transformation, kind=self.preprocessing)
        if self.preprocessing != 'no':
            self.training_set = ChallengeResponseSet(
                challenges=preprocess(self.training_set.challenges, self.k),
                responses=self.training_set.responses,
            )
            self.validation_set = ChallengeResponseSet(
                challenges=preprocess(self.validation_set.challenges, self.k),
                responses=self.validation_set.responses
            )
            if self.preprocessing == 'full':
                in_shape = self.k * self.n
            self.training_set.challenges = reshape(self.training_set.challenges, (self.training_set.N, in_shape))
            self.validation_set.challenges = reshape(self.validation_set.challenges, (self.validation_set.N, in_shape))
        if self.domain_in == 0:
            self.training_set.challenges = (self.training_set.challenges + 1) / 2
            self.validation_set.challenges = (self.validation_set.challenges + 1) / 2
        if self.domain_out == 0:
            self.validation_set.responses = (self.validation_set.responses + 1) / 2
            self.training_set.responses = (self.training_set.responses + 1) / 2

        l2_loss = l2(self.penalty) if self.penalty != 0 else None
        self.nn = Sequential()
        self.nn.add(Dense(
            units=self.layers[0],
            activation=self.activation,
            input_dim=in_shape,
            use_bias=True,
        ))
        if len(self.layers) > 1:
            for nodes in self.layers[1:]:
                self.nn.add(Dense(units=nodes, activation=self.activation, kernel_regularizer=l2_loss, use_bias=True))
        self.nn.add(Dense(units=1, activation='sigmoid' if self.domain_out == 0 else 'tanh', use_bias=True))

        if self.domain_out == 0:
            def pypuf_accuracy(y_true, y_pred):
                return 1 - tf_mean(tf_abs(y_true - y_pred))

            def log_loss(y_true, y_pred):
                return - multiply(y_true, log(y_pred + self.EPSILON)) \
                       - multiply((1 - y_true), log(1 - y_pred + self.EPSILON))

        elif self.domain_out == -1:
            def pypuf_accuracy(y_true, y_pred):
                return (1 + tf_mean(y_true * y_pred)) / 2

            def log_loss(y_true, y_pred):
                y_true = (y_true + 1) / 2
                y_pred = (y_pred + 1) / 2
                return - multiply(y_true, log(y_pred + self.EPSILON)) \
                       - multiply((1 - y_true), log(1 - y_pred + self.EPSILON))

        else:
            raise(Exception('The parameter "domain_out" has to be 0 or -1!'))

        def squared_hinge_loss_0_1(y_true, y_pred):
            return tf_max(0 * y_true, (1 - ((y_true * 2 - 1) * (y_pred * 2 - 1))) ** 2)

        self.nn.compile(
            optimizer=Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.EPSILON),
            loss=log_loss if self.loss == 'log_loss'
            else squared_hinge_loss_0_1 if self.loss == 'squared_hinge' and self.domain_out == 0
            else 'squared_hinge' if self.loss == 'squared_hinge' and self.domain_out == -1
            else 'binary_crossentropy',
            metrics=[pypuf_accuracy],
        )

        class Model:
            """
            This defines a wrapper for the learning model to fit in with the Learner framework.
            """
            def __init__(self, nn, n, k, preprocess, domain_in, domain_out):
                self.nn = nn
                self.n = n
                self.k = k
                self.preprocess = preprocess
                self.domain_in = domain_in
                self.domain_out = domain_out

            def eval(self, cs):
                cs_preprocessed = self.preprocess(challenges=cs, k=self.k)
                challenges = cs_preprocessed if self.domain_in == -1 else (cs_preprocessed + 1) / 2
                predictions = self.nn.predict(x=challenges)
                predictions_1_1 = predictions if self.domain_out == -1 else predictions * 2 - 1
                return sign(predictions_1_1).flatten()

        self.model = Model(
            nn=self.nn,
            n=self.n,
            k=self.k,
            preprocess=preprocess,
            domain_in=self.domain_in,
            domain_out=self.domain_out,
        )

    def learn(self):
        """
        Train the model with early stopping.
        """

        class DelayedEarlyStopping(EarlyStopping):

            def __init__(self, delay, tolerance, **kwargs):
                self.delay = delay
                self.tolerance = tolerance
                self.counter = 0
                self.current = 0
                self.highest = 0
                super().__init__(**kwargs)

            def on_epoch_end(self, epoch, logs=None):
                if epoch > self.delay:
                    value = logs.get(self.monitor)
                    if value > self.highest:
                        self.highest = value
                        if value >= self.current + self.tolerance:
                            self.current = self.highest
                            self.counter = 0
                            return
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.model.stop_training = True

        converged = DelayedEarlyStopping(
            delay=self.k,
            tolerance=self.tolerance,
            patience=self.patience,
            monitor='val_pypuf_accuracy',
            verbose=self.print_learning,
        )
        callbacks = [converged]
        self.history = self.nn.fit(
            x=self.training_set.challenges,
            y=self.training_set.responses,
            batch_size=self.batch_size,
            epochs=self.iteration_limit,
            callbacks=callbacks,
            validation_data=(self.validation_set.challenges, self.validation_set.responses),
            shuffle=True,
            verbose=self.print_learning,
        )
        return self.model
