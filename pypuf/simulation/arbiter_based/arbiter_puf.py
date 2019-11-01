"""
Collection of important Arbiter PUF variations.
"""
from numpy import concatenate
from numpy.random.mtrand import RandomState

from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.simulation.base import Simulation


class XORArbiterPUF(NoisyLTFArray):
    """
    XOR Arbiter PUF. k Arbiter PUFs (so-called chains) are evaluated in parallel, the individual results are XORed and
    then returned.
    Devadas, S.: Physical unclonable functions (PUFS) and secure processors. In: Workshop on Cryptographic Hardware and
    Embedded Systems (2009)
    """

    def __init__(self, n: int, k: int, seed: int = None, transform=None, noisiness=0, noise_seed=None):
        random_instance = RandomState(seed=seed) if seed is not None else RandomState()
        super().__init__(
            weight_array=self.normal_weights(n=n, k=k, random_instance=random_instance),
            transform=transform or LTFArray.transform_atf,
            combiner=LTFArray.combiner_xor,
            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(
                n=n,
                sigma_weight=1,
                noisiness=noisiness,
            ),
            random_instance=RandomState(seed=noise_seed) if noise_seed else RandomState()
        )
        self.n = n
        self.k = k

    def challenge_length(self) -> int:
        return self.n

    def response_length(self) -> int:
        return 1


class LightweightSecurePUF(XORArbiterPUF):
    """
    Lightweight Secure PUF. The challenge is transformed using a distinct pattern before the underlying XOR Arbiter PUF
    is evaluated on the transformed challenge. The result is returned.
    M. Majzoobi, F. Koushanfar, and M. Potkonjak, "Lightweight secure pufs", in IEEE/ACM International Conference on
    Computer-Aided Design (ICCAD 2008).
    """

    def __init__(self, n: int, k: int, seed: int = None, noisiness=0):
        super().__init__(n, k, seed, self.transform_lightweight_secure, noisiness)


class InterposePUF(Simulation):
    """
    Interpose PUF. Essentially consisting of two XOR Arbiter PUFs, where the second XOR Arbiter PUF has challenge length
    n+1. The value of the middle challenge bit is the result bit of the first XOR Arbiter PUF.
    Phuong Ha Nguyen, Durga Prasad Sahoo, Chenglu Jin, Kaleel Mahmood, Ulrich Rührmair and Marten van Dijk,
    "The Interpose PUF: Secure PUF Design against State-of-the-art Machine Learning Attacks", CHES 2019.
    """

    def __init__(self, n: int, k_down: int, k_up: int = 1, interpose_pos: int = None, seed: int = None, transform=None,
                 noisiness=0, noise_seed=None):
        super().__init__()
        self.n = n
        self.up = XORArbiterPUF(n, k_up, seed, transform, noisiness, noise_seed)
        self.down = XORArbiterPUF(n + 1, k_down, seed + 1 if seed is not None else None, transform, noisiness,
                                  noise_seed + 1 if noise_seed is not None else 0)
        self.interpose_pos = interpose_pos or n // 2

    def challenge_length(self) -> int:
        return self.up.challenge_length()

    def response_length(self) -> int:
        return self.down.response_length()

    def _interpose_bits(self, challenges):
        (N, _) = challenges.shape
        return self.up.eval(challenges).reshape(N, 1)

    def eval(self, challenges):
        (N, n) = challenges.shape
        interpose_bits = self._interpose_bits(challenges)
        down_challenges = concatenate(
            (challenges[:, :self.interpose_pos], interpose_bits, challenges[:, self.interpose_pos:]),
            axis=1
        )
        assert down_challenges.shape == (N, n + 1)
        return self.down.eval(down_challenges)
