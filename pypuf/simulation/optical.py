from numpy import ndarray
import numpy as np

from ..simulation import Simulation


class IntegratedOpticalPUF(Simulation):
    r"""
    Very basic simulation of an integrated optical PUF [RHUWDFJ13]_.
    This simulation is derived from the corresponding successful modeling attack [RHUWDFJ13]_.

    To compute a response to a given challenge :math:`c`, the simulation evaluates the value of

    .. math::
        \left| c \cdot A \cdot e^{\phi i} \right|^2,

    where :math:`A \in \mathbb{R}^{n \times m}` and :math:`\phi \in \mathbb{R}^{n \times m}`; :math:`n` is the
    challenge length, :math:`m` the response length.

    By default, the values for :math:`A` are chosen mutually independent and uniformly random in :math:`[0,1)`;
    the values for :math:`\phi` are chosen mutually independent and uniformly random in :math:`[0, 2\pi)`.

    .. warning::
        This simulation only gives a very rough, idealized idea of how integrated optical PUFs may behave.
        In particular, the :meth:`pypuf.metrics.uniqueness` may be quite different from the behavior real-world
        implementations. Also, the behavior of response pixels is mutually independent, which differs from behavior
        reported in the literature [RHUWDFJ13]_.
    """

    def __init__(self, n: int, m: int, seed: int) -> None:
        """
        Initializes a simulation for an Integrated Optical PUF with :math:`n` challenge bits, :math:`m` response bits,
        with randomness based on the given ``seed``.
        """
        super().__init__()
        self.n = n
        self.m = m

        A = np.random.default_rng(self.seed(f"intg. optical PUF {seed} A")).uniform(0, 1, size=(n, m))
        phi = np.random.default_rng(self.seed(f"intg. optical PUF {seed} phi")).uniform(0, 2 * np.pi, size=(n, m))
        self.T = A * np.exp(1j * phi)

    @property
    def challenge_length(self) -> int:
        return self.n

    @property
    def response_length(self) -> int:
        return self.m

    def eval(self, challenges: ndarray) -> ndarray:
        return np.abs(challenges @ self.T)**2
