import numpy as np
import pytest

import pypuf.simulation


@pytest.mark.parametrize('n', range(1, 12))
def test_ltf(n: int) -> None:
    weights = np.arange(n).reshape((1, n))
    bias = np.array([3])
    puf = pypuf.simulation.LTFArray(
        weight_array=weights,
        transform='id',
        bias=bias,
    )
    assert puf.val(np.array([[1] * n]))[0] == np.sum(weights) + bias
    assert puf.val(np.array([[-1] * n]))[0] == -np.sum(weights) + bias
