"""This module provides a simulation class which represents a dictator function."""
from pypuf.simulation.base import Simulation
from numpy import shape


class Dictator(Simulation):
    """This class represents a dictator function which means that only one input bit affects the response.
    As an example the function f(x) = x[i] is an i-th bit dictator."""

    def __init__(self, dictator):
        """
        :param dictator: int
                         Index of the dictator bit. Must be greater equal zero.
        """
        assert dictator >= 0, "The dictator index must be greater equal zero."
        self.dictator = dictator

    def eval(self, inputs):
        """
        :param inputs: array of pypuf.tools.RESULT_TYPE shape(N,n)
                       Array of challenges which should be evaluated by the simulation.
                       The dictator index must lie in the closed interval zero and n-1.

        :return: array of pypuf.tools.RESULT_TYPE shape(N)
                 Array of responses for the N different challenges.
        """
        assert 0 <= self.dictator < shape(inputs)[
            1], "Only challenges with number of bits greater {} are excepted".format(
            self.dictator
        )
        return inputs[:, self.dictator]
