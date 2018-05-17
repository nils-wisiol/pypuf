"""This module provides a bent function simulation class based on inner product modulo two."""
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.simulation.base import Simulation


class BentFunctionIpMod2(Simulation):
    """This class implements the inner product modulo two bent function"""
    def __init__(self, n):
        """
        :param n: int
                  An even Number of input bits.
        """
        assert n % 2 == 0, 'IP mod 2 is only defined for even n.'
        self.n = n

    def eval(self, inputs):
        """
        This function evaluates the inputs like inner product modulo two does.
        :param inputs: array of pypuf.tools.RESULT_TYPE shape(N,n)
                       Array of challenges which should be evaluated by the simulation.
        :return: array of pypuf.tools.RESULT_TYPE shape(N)
                 Array of responses for the N different challenges.
        """
        return LTFArray.combiner_ip_mod2(inputs)
