"""
This module provides a learning algorithm for Arbiter PUFs using a Multilayer Perceptron and Adam from Tensorflow.
This approach is based on the work of Aseeri et. al. "A Machine Learning-based Security Vulnerability Study on XOR PUFs
for Resource-Constraint Internet of Things", 2018 IEEE International Congress on Internet of Things (ICIOT),
San Francisco, CA, 2018, pp. 49-56.
"""

from os import environ
from numpy import sign
from numpy.random import seed
from numpy.random.mtrand import RandomState
from tensorflow import set_random_seed, ConfigProto, get_default_graph, Session, multiply, split
from tensorflow.python.platform.tf_logging import set_verbosity
from tensorflow.python.training.tensorboard_logging import ERROR
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.backend import mean as tf_mean, set_session, log, tanh
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense, Multiply
from tensorflow.python.keras.models import Model as Model_keras
from tensorflow.python.keras.optimizers import Adam
from pypuf.learner.base import Learner
from pypuf.tools import ChallengeResponseSet


environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_verbosity(ERROR)


class PureStochasticGradientDescent(Learner):
    """
    Define Learner that generates a Multilayer Perceptron model which uses the Adam optimization method.
    """

    SEED_RANGE = 2 ** 32
    EPSILON = 1e-6  # should not be smaller, else NaN in log
    DELAY = 5

    def __init__(self, n, k, training_set, validation_set, transformation, loss='log_loss', learning_rate=0.001,
                 penalty=0.0002, beta_1=0.9, beta_2=0.999, tolerance=0.0025, patience=4, print_learning=False,
                 iteration_limit=40, batch_size=1000, seed_model=0xc0ffee):
        """
        :param n:               int; positive, length of PUF (typically power of 2)
        :param k:               int; positive, width of PUF (typically between 1 and 8)
        :param training_set:    object holding two arrays (challenges, responses); elements are -1 and 1
        :param validation_set:  same as training_set, but only used for validation (not for training)
        :param transformation:  function; changes the challenges (while expanding them to the size of k * n)
        :param loss:            string: 'log_loss' or 'squared_hinge'; loss function used in learning
        :param learning_rate:   float; between 0 and 1, parameter of Adam optimizer
        :param penalty:         float; between 0 and 1, parameter of l2 regularization term used in learning
        :param beta_1:          float; between 0 and 1, parameter of Adam optimizer
        :param beta_2:          float; between 0 and 1, parameter of Adam optimizer
        :param tolerance:       float; between 0 and 1, stopping criterion: minimum change of loss
        :param patience:        int; positive, stopping criterion: maximum number of epochs with low change of loss
        :param iteration_limit: int; positive, maximum number of epochs
        :param batch_size:      int; between 1 and N, number of training samples used until updating the model's weights
        :param seed_model:      int; between 1 and 2**32, seed for PRNG
        :param print_learning:  bool; whether to print learning progress of Tensorflow
        """
        self.n = n
        self.k = k
        self.training_set = training_set
        self.validation_set = validation_set
        self.transformation = transformation
        self.loss = loss
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
        self.training_set = ChallengeResponseSet(
            challenges=self.transformation(self.training_set.challenges, self.k),
            responses=self.training_set.responses,
        )
        self.validation_set = ChallengeResponseSet(
            challenges=self.transformation(self.validation_set.challenges, self.k),
            responses=self.validation_set.responses
        )

        self.nn = XORArbiterPUF(k=self.k, penalty=self.penalty)

        def pypuf_accuracy(y_true, y_pred):
            return (1 + tf_mean(y_true * y_pred)) / 2

        def log_loss(y_true, y_pred):
            y_true = (y_true + 1) / 2
            y_pred = (y_pred + 1) / 2
            return - multiply(y_true, log(y_pred + self.EPSILON)) \
                   - multiply((1 - y_true), log(1 - y_pred + self.EPSILON))

        self.nn.compile(
            optimizer=Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.EPSILON),
            loss=log_loss if self.loss == 'log_loss'
            else 'squared_hinge' if self.loss == 'squared_hinge'
            else 'binary_crossentropy',
            metrics=[pypuf_accuracy],
        )

        class Model:
            """
            This defines a wrapper for the learning model to fit in with the Learner framework.
            """
            def __init__(self, nn, n, k, transformation):
                self.nn = nn
                self.n = n
                self.k = k
                self.transformation = transformation

            def eval(self, cs):
                return sign(self.nn.predict(x=self.transformation(challenges=cs, k=self.k))).flatten()

        self.model = Model(
            nn=self.nn,
            n=self.n,
            k=self.k,
            transformation=self.transformation,
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
            delay=self.DELAY,
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


class XORArbiterPUF(Model_keras):

    def __init__(self, k=1, n=64, penalty=0.0002):
        super(XORArbiterPUF, self).__init__()
        self.k = k
        self.n = n
        self.arbiter_chains = {}
        for i in range(k):
            self.arbiter_chains[i] = Dense(
                units=1,
                activation=None,
                kernel_regularizer=l2(penalty) if penalty != 0 else None,
                use_bias=True
            )
        self.xor = Multiply()
        self.activation = tanh

    def call(self, inputs, **kwargs):
        if self.k > 1:
            inputs = split(value=inputs, num_or_size_splits=self.k, axis=1)
        outputs = []
        for i in range(self.k):
            outputs.append(self.arbiter_chains[i](inputs[i]))
        return self.activation(self.xor(outputs)) if self.k > 1 else self.activation(outputs)
