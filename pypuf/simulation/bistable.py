import numpy as np

from .base import LTFArray


class XORBistableRingPUF(LTFArray):
    """
    A simulation of the XOR Bistable Ring PUF [XRHB15]_.

    The simulation is based on the findings by Xu et al. [XRHB15]_ which show that Bistable Ring PUFs can be modeled
    by linear threshold functions.
    """

    def __init__(self, n: int, k: int, weights: np.ndarray) -> None:
        """
        Initializes a XOR Bistable Ring PUF simulation.
        :param n: Number of challenge bits.
        :param k: Number of Bistable Ring PUFs
        :param weights: Array of weight values used for simulation.
        """
        if weights.shape != (k, n + 1):
            raise ValueError(f"Weights must be given as an array of shape (k, n+1) = ({k}, {n+1}), but {weights.shape}"
                             f"was given.")
        super().__init__(
            weight_array=weights[:, :-1],
            bias=weights[:, -1],
            transform=LTFArray.transform_id,
            combiner=self.combiner_xor,
        )


class BistableRingPUF(XORBistableRingPUF):
    """
    A simulation of the Bistable Ring PUF [CCLSR11]_.

    The simulation is based on the findings by Xu et al. [XRHB15]_ which show that Bistable Ring PUFs can be modeled
    by linear threshold functions.
    """

    def __init__(self, n: int, weights: np.ndarray) -> None:
        if weights.shape != (n + 1,):
            raise ValueError(f"Weights must be given as an array of length n+1 = {n+1}, but an array of shape "
                             f"{weights.shape} was given.")
        super().__init__(n, 1, weights.reshape((1, -1)))
