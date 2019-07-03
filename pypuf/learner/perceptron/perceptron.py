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
from tensorflow.python.keras.backend import maximum, mean, sign
import numpy as np

from functools import reduce


class MonomialFactory():
    """
        Collection of functions to build monomials.
        Currently only k-XOR Arbiter PUF monomials are supported.
    """
    def to_index_notation(self, mon):
        res = []
        for s in mon:
            if len(s) > 0:
                res.append(list(s))
        return res

    def monomials_atf(self, n):
        # Dict of set(indices) : coefficient
        return {frozenset(range(i,n)): 1 for i in range(n)}

    def multiply_sums(self, m1, m2):
        from collections import defaultdict
        res = defaultdict(int)
        for mon1, count1 in m1.items():
            for mon2, count2 in m2.items():
                 s3 = mon1.symmetric_difference(mon2)
                 res[s3] += count1 * count2
        return res

    def get_xor_arbiter_monomials(self, n, k):
        base_mon = self.monomials_atf(n)
        # Compute base_mon**k
        z = self.monomials_atf(n)
        for _ in range(k-1):
            z = self.multiply_sums(z, base_mon)
        return self.to_index_notation(z)




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
        # Multiply all values in the iterable
        self.multiply = lambda values: reduce(lambda x, y: x*y, values)

    def chal_to_xs(self, chal):
        """
        Convert challenge to Xs according to self.monomials
        """
        def mono_to_chalbits(mono, chal):
            """
            Map indices in monomial to challenge-bits
            """
            return map(lambda i: chal[i], mono)

        Xs = [self.multiply(mono_to_chalbits(mono, chal))
              for mono in self.monomials]
        return Xs


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
        self.linearize = lambda C: np.apply_along_axis(linearizer.chal_to_xs, 1, C)

        # Debugging data
        self.history = None

    def prepare(self):
        """
        Construct and compile Perceptron.
        Called in self.learn().
        """
        def pypuf_accuracy(y_true, y_pred):
            accuracy = (1 + mean(sign(y_true * y_pred))) / 2
            return maximum(accuracy, 1 - accuracy)
        model = Sequential()
        model.add(Dense(1, input_dim=self.input_len))
        model.add(Activation('tanh'))
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
