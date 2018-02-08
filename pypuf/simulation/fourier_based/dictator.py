"""This module provides a simulation class which represents a dictator function."""
from pypuf.simulation.base import Simulation
from numpy import shape


class Dictator(Simulation):
    """This class represents a dictator function which means that only one input bit affects the response.
    As an example the function f(x) = x[i] is an i-th bit dictator."""

    def __init__(self, dictator, n):
        """
        :param dictator: int
                         Index of the dictator bit. Must be greater equal zero.
        :param n: int
                  Number of input bits.
        """
        assert 0 <= dictator < n, "The dictator index must be between 0 and {}".format(n)
        self.dictator = dictator
        self.n = n

    def eval(self, inputs):
        """
        :param inputs: array of pypuf.tools.RESULT_TYPE shape(N,n)
                       Array of challenges which should be evaluated by the simulation.
                       The dictator index must lie in the closed interval zero and n-1.

        :return: array of pypuf.tools.RESULT_TYPE shape(N)
                 Array of responses for the N different challenges.
        """
        n = shape(inputs)[1]
        assert  n == self.n, "The number of input bits {} does not match {}".format(n, self.n)
        return inputs[:, self.dictator]
