from pypuf.learner.base import Learner          # Perceptron super class
from pypuf.simulation.base import Simulation    # Perceptron return type

# ML Utilities
from tensorflow.python.keras.models import Sequential
import numpy as np

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
        self.multiply = lambda values: reduce(lambda x,y: x*y, values)

    def chal_to_xs(self, chal):
        """
        Convert challenge to Xs according to self.monomials
        """
        mono_to_chalbits = lambda mono, chal: map(lambda i: chal[i], mono)
        Xs = [self.multiply(mono_to_chalbits(mono,chal))
              for mono in self.monomials]
        return Xs

class Perceptron(Learner):

    def __init__(self, train_set, valid_set, monomials=None,
                 batch_size=64, epochsgpu_id=None):
        """
        :param train_set: tools.TrainingSet
         Collection of Challenge/Response pairs to train the Perceptron
        :param valid_set: tools.TrainingSet
         Collection of Challenge/Response pairs to test the Perceptron
        :param gpu_id: int
         Indicates on which GPU the Perceptron will be learned
        """
        # Training parameters
        self.train_set = train_set
        self.valid_set = valid_set
        self.batch_size = batch_size
        self.epochs = epochs

        # If no monomials are provided, use identity
        if monomials is None:
            monomials = [[i] for i range(train_set.instance.n)]
        self.monomials = monomials
        self.input_len = len(self.monomials)

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
        model = Sequential()
        model.add(Dense(1, input_dim=self.input_len))
        model.add(Activation('tanh'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
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
                                      epochs=self.epochs
                                      validation_data=(x_valid,y_valid),
                                      verbose=True)

    def learn(self):
        """
        Fit the Perceptron and return a Simulation class that offers
        a 'eval(challenges)' method.
        """
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
        sim = type('PerceptronSimulation', (Simulation,))
        sim.eval = evaluate
        return sim


