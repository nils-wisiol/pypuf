from numpy import array
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.simulation.base import Simulation


class BentFunction(Simulation):

    def __init__(self, n):
        self.n = n


    def eval(self, inputs):
        return LTFArray.combiner_ip_mod2(inputs)