from numpy.random.mtrand import RandomState
from numpy import sign

from pypuf.simulation.base import Simulation


class NoisySRAM(Simulation):

    def __init__(self, size, noise, seed):
        self._seed = seed
        self._prng = RandomState(seed)
        self._size = size
        self._values = self._prng.normal(size=size)
        self.noise = noise

    def eval(self):
        noise = self._prng.normal(scale=self.noise, size=self._size)
        return sign(self._values + noise)

    def __str__(self):
        return 'NoisySRAM_%i_%f' % (self._seed, self.noise)
