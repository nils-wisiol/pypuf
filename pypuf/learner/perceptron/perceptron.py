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
    def __init__(self):
        self.DB = {}

    def to_index_notation(self, mon):
        return [list(s) for s in mon if len(s) > 0]

    def monomials_atf(self, n):
        # Dict of set(indices) : coefficient
        return {frozenset(range(i,n)): 1 for i in range(n)}

    def multiply_sums(self, s1, s2):
        return {m1.symmetric_difference(m2):c1*c2 for m1,c1 in s1.items()
                                                  for m2,c2 in s2.items()}

    def monomials_exp(self, mono, k):
        # Return monomials for k=1 and update DB if necessary
        if k == 1:
            return mono

        # k is not computed yet -> split in two halves
        k_star = k // 2
        A = self.monomials_exp(mono, k_star)
        res = self.multiply_sums(A, A)

        # If k was uneven, we need to multiply with the base monomial again
        if k % 2 == 1:
            res = self.multiply_sums(res, mono)
        return res

    def get_xor_arbiter_monomials(self, n, k):
        mono = self.monomials_atf(n)
        res = self.monomials_exp(mono, k)
        return self.to_index_notation(res)

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
