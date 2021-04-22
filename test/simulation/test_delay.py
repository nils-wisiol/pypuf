import numpy as np
import pytest

import pypuf.simulation
import pypuf.metrics


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


@pytest.mark.parametrize('n', [32, 64, 128])
@pytest.mark.parametrize('k', [1, 2, 4])
def test_ff_degenerated(n: int, k: int) -> None:
    seed = 1

    ff_puf = pypuf.simulation.FeedForwardArbiterPUF(
        n=n, ff=[], seed=seed,
    )
    puf = pypuf.simulation.LTFArray(
        weight_array=pypuf.simulation.LTFArray.normal_weights(
            n=n, k=1, seed=pypuf.simulation.LTFArray.seed(f'FeedForwardArbiterPUF {seed} weights')
        ),
        transform=pypuf.simulation.XORArbiterPUF.transform_atf,
    )

    assert pypuf.metrics.similarity(puf, ff_puf, seed=1) == 1


def test_ff_1_loop() -> None:
    ff_puf = pypuf.simulation.FeedForwardArbiterPUF(
        n=4, ff=[(3, 4)], seed=None,
    )
    assert ff_puf.weight_array.shape == (1, 4 + 1 + 1)
    ff_puf.weight_array[0] = [1, 1, 1, 100, 1, 0]

    # challenge: [ 1  1  1  1 XX]
    # loop challenge [ 1  1  1]
    # loop features  [ 1  1  1] => 3 => 1
    # extended challenge: [ 1  1  1  1  1]
    # features: [ 1  1  1  1  1]
    assert ff_puf.val(np.array([[1, 1, 1, 1]]))[0] == 104

    # challenge: [1 -1 -1  1]
    # loop challenge: [ 1 -1 -1]
    # transformed loop challenge [ 1  1 -1] => 2 => 1
    # extended challenge [ 1 -1 -1  1  1]
    # transformed challenge [ 1  1 -1  1  1] => 102
    assert ff_puf.val(np.array([[1, -1, -1, 1]]))[0] == 102


def test_ff_2_loops_sequentially() -> None:
    ff_puf = pypuf.simulation.FeedForwardArbiterPUF(
        n=4, ff=[(2, 3), (3, 4)], seed=None,
    )
    assert ff_puf.weight_array.shape == (1, 4 + 2 + 1)
    ff_puf.weight_array[0] = [1, 1, 1, 1, 1, 1, 0]

    #              0  1  2  3  4  5
    # challenge: [ 1 -1 -1 XX XX  1]
    # challenge to loop 1: [ 1 -1]
    # features for loop 1: [-1 -1] => dd -2 => sign -1
    # challenge: [ 1 -1 -1 -1 XX  1]
    # challenge to loop 2: [ 1 -1 -1]
    # features for loop 2: [ 1  1 -1] => dd 2 => sign 1
    # challenge: [ 1 -1 -1 -1  1  1]
    # features:  [-1 -1  1 -1  1  1] => dd 0
    assert ff_puf.val(np.array([[1, -1, -1, 1]]))[0] == 0


def test_ff_2_loops_interleaved() -> None:
    ff_puf = pypuf.simulation.FeedForwardArbiterPUF(
        n=4, ff=[(1, 3), (2, 4)], seed=None,
    )
    assert ff_puf.weight_array.shape == (1, 4 + 2 + 1)
    ff_puf.weight_array[0] = [1, 1, 1, 1, 1, 1, 0]

    #              0  1  2  3  4  5
    # challenge: [ 1 -1 -1 XX XX  1]
    # challenge to loop 1: [ 1 -1 -1]
    # features for loop 1: [ 1  1 -1] => dd 1 => sign 1
    # challenge: [ 1 -1 -1  1 XX  1]
    # challenge to loop 2: [ 1 -1]
    # features for loop 2: [-1 -1] => dd -2 => sign -1
    # challenge: [ 1 -1 -1  1 -1  1]
    # features:  [-1 -1  1 -1 -1  1] => dd -2
    assert ff_puf.val(np.array([[1, -1, -1, 1]]))[0] == -2


def test_ff_2_loops_same_arbiter() -> None:
    ff_puf = pypuf.simulation.FeedForwardArbiterPUF(
        n=4, ff=[(1, 3), (1, 4)], seed=None,
    )
    assert ff_puf.weight_array.shape == (1, 4 + 2 + 1)
    ff_puf.weight_array[0] = [1, 1, 1, 1, 1, 1, 0]

    #              0  1  2  3  4  5
    # challenge: [ 1 -1 -1 XX XX  1]
    # challenge to loop 1: [ 1]
    # features for loop 1: [ 1] => dd 1 => sign 1
    # challenge: [ 1 -1 -1  1  1  1]
    # features:  [ 1  1 -1  1  1  1] => dd 5
    assert ff_puf.val(np.array([[1, -1, -1, 1]]))[0] == 4


def test_ff_2_loops_homogeneous() -> None:
    # based on test_ff_2_loops_interleaved
    ff_puf = pypuf.simulation.XORFeedForwardArbiterPUF(
        n=4, k=2, ff=[(1, 3), (2, 4)], seed=None,
    )
    assert ff_puf.simulations[0].weight_array.shape == (1, 4 + 2 + 1)
    assert ff_puf.simulations[1].weight_array.shape == (1, 4 + 2 + 1)
    ff_puf.simulations[0].weight_array[0] = [1, 1, 1, 1, 1, 1, 0]
    ff_puf.simulations[1].weight_array[0] = [1, 1, 1, 1, 1, 1, 0]
    assert ff_puf.val(np.array([[1, -1, -1, 1]]))[0] == (-2)**2


def test_ff_2_loops_inhomogeneous() -> None:
    ff_puf = pypuf.simulation.XORFeedForwardArbiterPUF(
        n=4, k=2, ff=[
            [(1, 3), (1, 4)],
            [(1, 3), (2, 4)],
        ], seed=None,
    )
    assert ff_puf.simulations[0].weight_array.shape == (1, 4 + 2 + 1)
    assert ff_puf.simulations[1].weight_array.shape == (1, 4 + 2 + 1)
    ff_puf.simulations[0].weight_array[0] = [1, 1, 1, 1, 1, 1, 0]
    ff_puf.simulations[1].weight_array[0] = [1, 1, 1, 1, 1, 1, 0]
    assert ff_puf.val(np.array([[1, -1, -1, 1]]))[0] == -2 * 4


def test_ff_reproducible_noise() -> None:
    pufs = [
        pypuf.simulation.XORFeedForwardArbiterPUF(
            n=4, k=2, ff=[
                [(1, 3), (1, 4)],
                [(1, 3), (2, 4)],
            ],
            seed=123,
            noisiness=.25,
        )
        for _ in range(2)
    ]
    assert pypuf.metrics.similarity(pufs[0], pufs[1], seed=1) == 1
