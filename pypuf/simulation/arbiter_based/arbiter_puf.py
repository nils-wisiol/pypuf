"""
Collection of important Arbiter PUF variations.
"""
from numpy import concatenate
from numpy.random.mtrand import RandomState

from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.simulation.base import Simulation


class XORArbiterPUF(LTFArray):
    """
    XOR Arbiter PUF. k Arbiter PUFs (so-called chains) are evaluated in parallel, the individual results are XORed and
    then returned.
    Devadas, S.: Physical unclonable functions (PUFS) and secure processors. In: Workshop on Cryptographic Hardware and
    Embedded Systems (2009)
    """

    def __init__(self, n: int, k: int, seed: int = None):
        random_instance = RandomState(seed=seed) if seed is not None else RandomState()
        super().__init__(
            weight_array=self.normal_weights(n=n, k=k, random_instance=random_instance),
            transform=LTFArray.transform_atf,
            combiner=LTFArray.combiner_xor,
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

    def __init__(self, n: int, k: int, seed: int = None):
        super().__init__(n, k, seed)
        self.transform = self.transform_lightweight_secure


class InterposePUF(Simulation):
    """
    Interpose PUF. Essentially consisting of two XOR Arbiter PUFs, where the second XOR Arbiter PUF has challenge length
    n+1. The value of the middle challenge bit is the result bit of the first XOR Arbiter PUF.
    Phuong Ha Nguyen, Durga Prasad Sahoo, Chenglu Jin, Kaleel Mahmood, Ulrich Rührmair and Marten van Dijk,
    "The Interpose PUF: Secure PUF Design against State-of-the-art Machine Learning Attacks", CHES 2019.
    """

    def __init__(self, n: int, k: int, k_up: int = 1, interpose_pos: int = None, seed: int = None):
        super().__init__(n, k_up, seed)
        self.up = XORArbiterPUF(n, k, seed)
        self.down = XORArbiterPUF(n + 1, k, seed + 1)
        self.interpose_pos = interpose_pos or n // 2

    def challenge_length(self) -> int:
        return self.up.challenge_length()

    def response_length(self) -> int:
        return self.down.response_length()

    def eval(self, challenges):
        (N, n) = challenges.shape
        interpose_bits = self.up.eval(challenges).reshape(N, 1)
        down_challenges = concatenate(
            (challenges[:, :self.interpose_pos], interpose_bits, challenges[:, self.interpose_pos:]),
            axis=1
        )
        assert down_challenges.shape == (N, n + 1)
        return self.down.eval(down_challenges)
