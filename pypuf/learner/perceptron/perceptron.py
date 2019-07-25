"""
This module implements a pypuf Perceptron Learner.
The Learner is capable to be instatiated with 'monomials' which define
how to convert challenges into a different feature space.
"""
from pypuf.learner.base import Learner          # Perceptron super class
from pypuf.simulation.base import Simulation    # Perceptron return type

# ML Utilities
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.backend import maximum, mean, sign, exp
import numpy as np

from pypuf.bipoly import BiPoly

class LinearizationModel():
    """
    Helper class to linearize challenges of a k-XOR PUF.
    Instantiate using 'monomials' - a list of lists containing indices,
    which defines how to compute linearized variables from a challenge.
    Example: [[1,2,4],[1,6]] => X1 = C1*C2*C4; X2 = C1*C6
    """

    def __init__(self, monomials):
        """
        :param monomials: list of lists containing indices to compute x's
                          from challenge
        """
        # Monomials, defining how to build Xs from a challenge
        self.monomials = monomials

    def linearize(self, inputs):
        """
        Convert array of challenges to Xs accoring to self.monomials.
        Param inputs has shape N, n - meaning N challenges of n bits.
        """
        N, n = inputs.shape
        out = np.empty(shape=(N, len(self.monomials)), dtype=np.int8)
        for idx, m in enumerate(self.monomials):
            out[:, idx] = np.prod(inputs[:, list(m)], axis=1)
        return out

class Perceptron(Learner):
    """
    Perceptron Learner - learn() method returns a pypuf.simulation.base.Simulation
    object, containing a eval() method which can be used to predict challenges
    """

    def __init__(self, train_set, valid_set, monomials=None,
                 batch_size=64, epochs=1000, gpu_id=None):
        """
        :param train_set: tools.TrainingSet
         Collection of Challenge/Response pairs to train the Perceptron
        :param valid_set: tools.TrainingSet
         Collection of Challenge/Response pairs to test the Perceptron
        :param monomials: list of int lists
         Defines how to boost challenge-bits to feature space
        :param batch_size: int
         Batch learning size - forwarded to tensorflow model.fit()
        :param epochs: int
         Number of iterations to train perceptron - forwarded to
         tensorflow model.fit()
        :param gpu_id: int
         Indicates on which GPU the Perceptron will be learned
        """
        # Training parameters
        self.train_set = train_set
        self.valid_set = valid_set
        self.batch_size = batch_size
        self.epochs = epochs
        self.gpu_id = gpu_id

        # If no monomials are provided, use identity
        if monomials is None:
            monomials = [[i] for i in range(train_set.instance.n)]
        self.monomials = monomials

        # Model parameters
        self.input_len = len(self.monomials)
        self.model = None

        # Build linearization model
        linearizer = LinearizationModel(self.monomials)
        # Apply linearization to each row in numpy array
        self.linearize = linearizer.linearize

        # Debugging data
        self.history = None

    def prepare(self):
        """
        Construct and compile Perceptron.
        Called in self.learn().
        """
        def pypuf_accuracy(y_true, y_pred):
            accuracy = (1 + mean(sign(y_true * y_pred))) / 2
            return accuracy
        def soelter_d(y_true, y_pred):
            return 1 - (1/exp(-y_true*y_pred))
        model = Sequential()
        model.add(Dense(1, input_dim=self.input_len))
        model.add(Activation('softsign'))
        model.compile(loss='squared_hinge',
                      optimizer='adam',
                      metrics=[pypuf_accuracy])
        self.model = model

    def fit(self):
        """
        Train the Perceptron.
        Called in self.learn().
        """
        # Convert challenges to linearized subchallenge representation
        x = self.linearize(self.train_set.challenges)
        y = self.train_set.responses
        x_valid = self.linearize(self.valid_set.challenges)
        y_valid = self.valid_set.responses

        self.history = self.model.fit(x=x, y=y,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      validation_data=(x_valid, y_valid),
                                      verbose=False)

    def learn(self):
        """
        Fit the Perceptron and return a Simulation class that offers
        a 'eval(challenges)' method.
        """
        self.prepare()
        # Fit the Perceptron (self.model) normally or using GPU
        if self.gpu_id is None:
            self.fit()
        else:
            import tensorflow as tf
            with tf.device("/gpu:%d" % self.gpu_id):
                self.fit()

        # Build evaluation function
        def evaluate(chals):
            x = self.linearize(chals)
            y = self.model.predict(x)
            return np.sign(y)

        # Create Simulation object and return it
        sim = type('PerceptronSimulation', (Simulation,), {})
        sim.eval = evaluate
        sim.n = self.input_len
        return sim
