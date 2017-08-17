from pypuf.simulation.base import Simulation
from pypuf import tools
import numpy as np


class FourierCoefficient:
    def __init__(self, s, val):
        self.s = s
        self.val = val


class FourierExpansion(Simulation):
    """
    Boolean Function {-1,1}^n -> (real numbers) defined by its Fourier coefficients, given as an array of the form
    [ coefficient, coefficient, coefficient, ... ] where each coefficient has to have members coefficient.val and
    coefficient.s, val being the value and s being a {0,1}-array indicating the character.
    """

    def __init__(self, fourier_coefficients):
        self.fourier_coefficients = fourier_coefficients
        self.n = len(fourier_coefficients[0].s)

    def eval(self, inputs):
        vals = np.array(
            [coefficient.val * tools.chi_vectorized(coefficient.s, inputs)
             for coefficient in self.fourier_coefficients]
        ).T
        return np.sum(vals, axis=1)


class FourierExpansionSign(FourierExpansion):
    """
    Same as FourierExpansion, but only return the sign of the function value.
    Use val() to access the real number value.
    """

    def eval(self, inputs):
        return np.sign(super(FourierExpansionSign, self).eval(inputs))

    def val(self, inputs):
        return super(FourierExpansionSign, self).eval(inputs)
