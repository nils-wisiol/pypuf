"""
This module provides a model for the Fourier expansion of Boolean functions.
(http://www.contrib.andrew.cmu.edu/~ryanod/?p=207)
"""
import numpy as np
from pypuf.simulation.base import Simulation
from pypuf import tools


class FourierCoefficient:
    """
    This class represents a Fourier coefficient for a given set with
    a given value.
    """
    def __init__(self, s, val):
        """
        :param s: list of int
                  A {0,1}-array indicating the coefficient's index set
        :param val: float
                    The value of the coefficient as a real number.

        """
        self.s = s
        self.val = val


class FourierExpansion(Simulation):
    """
    Boolean Function {-1,1}^n -> (real numbers) defined by its Fourier coefficients.
    """

    def __init__(self, fourier_coefficients):
        """

        :param fourier_coefficients: list of FourierCoefficient
                                     The coefficients of the Fourier expansion.
        """
        self.fourier_coefficients = fourier_coefficients
        self.n = len(fourier_coefficients[0].s)

    def eval(self, challenges):
        """
        Evaluates a given array of inputs.
        :param challenges: array of int shape(N,n)
                       {-1,1}-valued inputs to be evaluated.
        :return: array of float
                 real valued responses
        """
        vals = np.array(
            [coefficient.val * tools.chi_vectorized(coefficient.s, challenges)
             for coefficient in self.fourier_coefficients]
        ).T
        return np.sum(vals, axis=1)


class FourierExpansionSign(FourierExpansion):
    """
    Same as FourierExpansion, but only return the sign of the function value.
    Use val() to access the real number value.
    """

    def eval(self, challenges):
        """
        Evaluates a given array of inputs.
        :param challenges: array of int shape(N,n)
                       {-1,1}-valued inputs to be evaluated.
        :return: array of float
                 {-1,1}-valued responses
        """
        return np.sign(super(FourierExpansionSign, self).eval(challenges))

    def val(self, challenges):
        """
        Evaluates a given array of inputs.
        :param challenges: array of int shape(N,n)
                       {-1,1}-valued inputs to be evaluated.
        :return: array of float
                 real valued responses
        """
        return super(FourierExpansionSign, self).eval(challenges)
