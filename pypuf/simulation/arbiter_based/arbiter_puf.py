from numpy import concatenate
from numpy.random.mtrand import RandomState

from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray


class XORArbiterPUF(NoisyLTFArray):

    def __init__(self, n: int, k: int, seed: int = None, transform=None, noisiness=0, noise_seed=None):
        random_instance = RandomState(seed=seed) if seed is not None else RandomState()
        super().__init__(
            weight_array=self.normal_weights(n=n, k=k, random_instance=random_instance),
            transform=transform or LTFArray.transform_id,
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

    def __init__(self, n: int, k: int, seed: int = None, noisiness=0):
        super().__init__(n, k, seed, self.transform_lightweight_secure, noisiness)


class InterposePUF(XORArbiterPUF):

    def __init__(self, n: int, k: int, k_up: int = 1, interpose_pos: int = None, seed: int = None, transform=None,
                 noisiness=0, noise_seed=None):
        super().__init__(n, k_up, seed, transform, noisiness, noise_seed)
        self.down = XORArbiterPUF(n + 1, k, seed + 1 if seed else None, transform, noisiness)
        self.interpose_pos = interpose_pos or n // 2

    def eval(self, challenges, **kwargs):
        (N, n) = challenges.shape
        interpose_bits = super().eval(challenges).reshape(N, 1)
        down_challenges = concatenate(
            (challenges[:, :self.interpose_pos], interpose_bits, challenges[:, self.interpose_pos:]),
            axis=1
        )
        assert down_challenges.shape == (N, n + 1)
        return self.down.eval(down_challenges)
