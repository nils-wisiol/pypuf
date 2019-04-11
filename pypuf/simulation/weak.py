from numpy.random.mtrand import RandomState
from numpy import sign, broadcast_to

from pypuf.simulation.base import Simulation


class NoisySRAM(Simulation):

    def __init__(self, size, noise, seed_skew, seed_noise):
        self._seed_noise = seed_noise
        self._seed_skew = seed_skew
        self.noise = noise
        self._noise_prng = RandomState(seed=seed_noise)
        self._size = size
        self._skew = RandomState(seed=seed_skew).normal(size=size)

    def challenge_length(self):
        return 0

    def response_length(self):
        return self._size

    def eval(self, challenges):
        (N, n) = challenges.shape
        assert n == 0

        # Same skew for every evaluation
        skew = broadcast_to(self._skew, (N, self._size))

        # Different noise for every evaluation
        noise = self._noise_prng.normal(scale=self.noise, size=(N, self._size))

        return sign(skew + noise)

    def __str__(self):
        return 'NoisySRAM_%i_%i_%f' % (self._seed_skew, self._seed_noise, self.noise)
