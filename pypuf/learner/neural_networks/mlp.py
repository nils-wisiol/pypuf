from os import environ, remove
from os.path import exists

from numpy import reshape, sign
from numpy.random import seed
from numpy.random.mtrand import RandomState
from tensorflow import set_random_seed, ConfigProto, get_default_graph, Session, TensorShape
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.platform.tf_logging import set_verbosity
from tensorflow.python.training.tensorboard_logging import ERROR
from tensorflow.python.keras.backend import maximum as tf_max, mean as tf_mean, sign as tf_sign, set_session
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Dense, dot, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

from pypuf.learner.base import Learner
from pypuf.tools import ChallengeResponseSet


SEED_RANGE = 2 ** 32
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_verbosity(ERROR)


class MultiLayerPerceptron(Learner):

    def __init__(self, log_name, n, k, training_set, validation_set, transformation=None,
                 print_keras=False, iteration_limit=1000, batch_size=1000, seed_model=None):
        self.log_name = log_name
        self.n = n
        self.k = k
        self.transformation = transformation
        self.training_set = training_set
        self.validation_set = validation_set
        self.print_keras = 0 if not print_keras else 1
        self.iteration_limit = iteration_limit
        self.batch_size = min(batch_size, training_set.N)
        self.seed_model = RandomState(seed_model).randint(SEED_RANGE)
        self.checkpoint = 'checkpoint.{}_{}_{}_{}_{}'.format(
            training_set.N, n, k, 'no_preprocess' if transformation is None else transformation.__name__,
            hex(self.seed_model)
        ) + '.hdf5'
        self.nn = None
        self.history = None
        self.model = None

    def prepare(self):
        seed(self.seed_model)
        set_random_seed(self.seed_model)
        session_conf = ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        set_session(Session(graph=get_default_graph(), config=session_conf))
        in_shape = self.n
        if self.transformation is not None:
            in_shape = self.k * self.n
            self.training_set = ChallengeResponseSet(
                challenges=self.transformation(self.training_set.challenges, self.k),
                responses=self.training_set.responses
            )
            self.validation_set = ChallengeResponseSet(
                challenges=self.transformation(self.validation_set.challenges, self.k),
                responses=self.validation_set.responses
            )
            self.training_set.challenges = reshape(self.training_set.challenges, (self.training_set.N, in_shape))
            self.validation_set.challenges = reshape(self.validation_set.challenges, (self.validation_set.N, in_shape))

        l2_loss = l2(0.01)
        self.nn = Sequential()
        self.nn.add(Dense(3 * 2 ** (self.k - 1), input_dim=in_shape, activation='relu', kernel_regularizer=l2_loss))
        self.nn.add(Dense(3 * 2 ** (self.k - 1), activation='relu', kernel_regularizer=l2_loss))
        self.nn.add(Dense(1, activation='tanh', activity_regularizer=l2_loss))
        """
        # Try LinearLayer
        self.nn = Sequential()
        self.nn.add(LinearFunction(input_shape=TensorShape([in_shape, 1])))
        self.nn.add(Dense(1, activation='tanh'))
        """

        def pypuf_accuracy(y_true, y_pred):
            accuracy = (1 + tf_mean(tf_sign(y_true * y_pred))) / 2
            return tf_max(accuracy, 1 - accuracy)

        self.nn.compile(
            optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True),
            loss='squared_hinge',
            metrics=[pypuf_accuracy]
        )

        class Model:
            def __init__(self, nn, n, k, transformation):
                self.nn = nn
                self.n = n
                self.k = k
                self.transformation = transformation

            def eval(self, cs):
                if self.transformation is not None:
                    cs = reshape(self.transformation(cs, self.k), (len(cs), self.k * self.n))
                return sign(self.nn.predict(cs)).flatten()

        self.model = Model(self.nn, self.n, self.k, self.transformation)

    def learn(self):
        class TerminateOnThreshold(EarlyStopping):
            def __init__(self, threshold, **kwargs):
                self.threshold = threshold
                super().__init__(**kwargs)

            def on_epoch_end(self, epoch, logs=None):
                if logs.get(self.monitor) >= self.threshold:
                    self.model.stop_training = True
                super().on_epoch_end(epoch, logs)

        checkpoint = ModelCheckpoint(
            filepath=self.checkpoint,
            verbose=self.print_keras,
            monitor='val_pypuf_accuracy',
            save_best_only=True,
            mode='max'
        )
        accurate = TerminateOnThreshold(
            threshold=1-(self.k*0.01),
            monitor='val_pypuf_accuracy',
            patience=self.iteration_limit,
            verbose=self.print_keras,
            mode='max',
            restore_best_weights=True
        )
        converged = EarlyStopping(
            monitor='val_pypuf_accuracy',
            min_delta=0.001,
            patience=(200 // self.k),
            verbose=self.print_keras,
            mode='max',
            restore_best_weights=True
        )
        callbacks = [checkpoint, accurate, converged]
        self.history = self.nn.fit(
            x=self.training_set.challenges,
            y=self.training_set.responses,
            batch_size=self.batch_size,
            epochs=self.iteration_limit,
            callbacks=callbacks,
            validation_data=(self.validation_set.challenges, self.validation_set.responses),
            shuffle=True,
            verbose=self.print_keras
        )
        if exists(self.checkpoint):
            remove(self.checkpoint)
        return self.model


"""
class LinearFunction(Layer):

    def __init__(self, **kwargs):
        self.output_dim = 1
        self.ltf_weights = None
        super(LinearFunction, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ltf_weights = self.add_weight(
            name='ltf_weights',
            shape=(input_shape, self.output_dim),
            initializer=RandomNormal(mean=0, stddev=1, seed=seed),
            trainable=True
        )
        super(LinearFunction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        return dot(inputs, self.ltf_weights)

    def compute_output_shape(self, input_shape):
        return self.output_dim
"""
