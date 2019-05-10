from os import environ

from numpy import reshape, sign
from numpy.random import seed
from numpy.random.mtrand import RandomState
from tensorflow import set_random_seed, ConfigProto, get_default_graph, Session, multiply
from tensorflow.python.platform.tf_logging import set_verbosity
from tensorflow.python.training.tensorboard_logging import ERROR
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.backend import maximum as tf_max, mean as tf_mean, sign as tf_sign, set_session, log
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

    SEED_RANGE = 2 ** 32
    EPSILON = 1e-7  # should not be smaller, else NaN in log

    def __init__(self, n, k, training_set, validation_set, transformation, preprocessing, layers=(10, 10),
                 activation='relu', zero_one=False, learning_rate=0.001, penalty=0.0001, beta_1=0.9, beta_2=0.999,
                 tolerance=0.001, patience=5, checkpoint_name=None, print_learning=False, termination_threshold=1.0,
                 iteration_limit=100, batch_size=1000, seed_model=0xc0ffee):
        self.n = n
        self.k = k
        self.training_set = training_set
        self.validation_set = validation_set
        self.transformation = transformation
        self.preprocessing = preprocessing
        self.layers = layers
        self.activation = activation
        self.zero_one = zero_one
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tolerance = tolerance
        self.patience = patience
        self.iteration_limit = iteration_limit
        self.batch_size = min(batch_size, training_set.N)
        self.seed_model = RandomState(seed_model).randint(self.SEED_RANGE)
        self.termination_threshold = termination_threshold
        self.checkpoint = 'checkpoint.{}_{}_{}_{}'.format(n, k, preprocessing, checkpoint_name) + '.hdf5'
        self.print_learning = 0 if not print_learning else 1
        self.nn = None
        self.history = None
        self.model = None

    def prepare(self):
        seed(self.seed_model)
        set_random_seed(self.seed_model)
        session_conf = ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        set_session(Session(graph=get_default_graph(), config=session_conf))
        in_shape = self.n
        preprocess = LTFArray.preprocess(transformation=self.transformation, kind=self.preprocessing)
        if self.preprocessing != 'no':
            self.training_set = ChallengeResponseSet(
                challenges=preprocess(self.training_set.challenges, self.k),
                responses=((self.training_set.responses + 1) / 2) if self.zero_one else self.training_set.responses
            )
            self.validation_set = ChallengeResponseSet(
                challenges=preprocess(self.validation_set.challenges, self.k),
                responses=((self.validation_set.responses + 1) / 2) if self.zero_one else self.validation_set.responses
            )
            if self.preprocessing == 'full':
                in_shape = self.k * self.n
            self.training_set.challenges = reshape(self.training_set.challenges, (self.training_set.N, in_shape))
            self.validation_set.challenges = reshape(self.validation_set.challenges, (self.validation_set.N, in_shape))
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
        self.nn.add(Dense(units=1, activation='sigmoid' if self.zero_one else 'tanh', use_bias=True))

        def pypuf_accuracy(y_true, y_pred):
            accuracy = tf_mean(tf_sign(y_true * y_pred))
            return tf_max(accuracy, 1 - accuracy)

        def pypuf_log_loss_1_1(y_true, y_pred):
            y_true = (y_true + 1) / 2
            y_pred = (y_pred + 1) / 2
            return - multiply(y_true, log(y_pred + self.EPSILON)) \
                   - multiply((1 - y_true), log(1 - y_pred + self.EPSILON))

        def pypuf_log_loss_0_1(y_true, y_pred):
            return - multiply(y_true, log(y_pred + self.EPSILON)) \
                   - multiply((1 - y_true), log(1 - y_pred + self.EPSILON))

        self.nn.compile(
            optimizer=Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.EPSILON),
            loss=pypuf_log_loss_0_1 if self.zero_one else pypuf_log_loss_1_1,
            metrics=[pypuf_accuracy],
        )

        class Model:
            def __init__(self, nn, n, k, preprocess, zero_one):
                self.nn = nn
                self.n = n
                self.k = k
                self.preprocess = preprocess
                self.zero_one = zero_one

            def eval(self, cs):
                predictions = self.nn.predict(x=self.preprocess(challenges=cs, k=self.k))
                predictions = (predictions * 2) - 1 if self.zero_one else predictions
                return sign(predictions).flatten()

        self.model = Model(nn=self.nn, n=self.n, k=self.k, preprocess=preprocess, zero_one=self.zero_one)

    def learn(self):
        converged = EarlyStopping(
            monitor='val_loss',
            min_delta=self.tolerance,
            patience=self.patience,
            verbose=self.print_learning,
            mode='min',
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
