import numpy as np
from numpy import ndarray

from pypuf.simulation import Simulation, XORPUF


def test_xor_puf() -> None:
    class ConstantSimulation(Simulation):

        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        @property
        def challenge_length(self) -> int:
            return 1

        @property
        def response_length(self) -> int:
            return 2

        def val(self, challenges: ndarray) -> ndarray:
            return np.array([(self.value, self.value)] * challenges.shape[0])

        def eval(self, challenges: ndarray) -> ndarray:
            return np.sign(self.val(challenges))

    puf = XORPUF([ConstantSimulation(3), ConstantSimulation(4)])
    assert (puf.val(np.array([[1]]))[0] == [12, 12]).all()
    assert (puf.eval(np.array([[1]]))[0] == [1, 1]).all()
