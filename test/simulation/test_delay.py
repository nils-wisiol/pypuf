import numpy as np
import pytest

import pypuf.simulation
import pypuf.metrics
import pypuf.io


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


@pytest.mark.parametrize('n', [64, 1000])
def test_ff_invalid_feed_point(n: int) -> None:
    puf = pypuf.simulation.FeedForwardArbiterPUF(n=n, ff=[(1, n+1)], seed=1)
    challenges = pypuf.io.random_inputs(n=n, N=1, seed=1)
    with pytest.raises(IndexError):
        puf.eval(challenges)


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


def test_ff_many_loops() -> None:
    puf = pypuf.simulation.XORFeedForwardArbiterPUF(
        n=64, k=1, ff=[(2, 30), (8, 58), (25, 32), (48, 51)], seed=1,
    )
    assert set(np.unique(puf.eval(pypuf.io.random_inputs(n=puf.challenge_length, N=100, seed=1)))) == {-1, 1}
    assert -.5 < pypuf.metrics.bias(puf, seed=1) < .5
    assert pypuf.metrics.bias(puf, seed=1) != 0


def beli_puf_debug_weights(n: int) -> np.ndarray:
    return np.array([
        10**(n // 2 - j - 1) * np.array(range(8))
        for j in range(n // 2)
    ]).reshape((1, n // 2, 8))


def zeros_except(n, exceptions):
    a = np.zeros(n, dtype=np.int8)
    a[exceptions] = 1
    return a


def test_beli_puf_delay() -> None:
    beli_puf = pypuf.simulation.BeliPUF(n=4, k=1, seed=1)
    beli_puf.delays = beli_puf_debug_weights(n=4)
    challenges = np.array([
        [-1, -1, -1, -1],  # all crossed
        [1, 1, 1, 1],  # all straight
        [-1, 1, -1, 1],  # top crossed, bottom straight
        [1, -1, 1, -1],  # top straight, bottom crossed
        [1, -1, 1, 1],
    ])

    paths = beli_puf.signal_path(challenges)
    # challenge with all crossed paths
    assert (paths[0, 0] == [6, 2]).all()
    assert (paths[0, 1] == [2, 3]).all()
    assert (paths[0, 2] == [7, 6]).all()
    assert (paths[0, 3] == [3, 7]).all()

    # challenge with all straight paths
    assert (paths[1, 0] == [0, 0]).all()
    assert (paths[1, 1] == [4, 1]).all()
    assert (paths[1, 2] == [1, 4]).all()
    assert (paths[1, 3] == [5, 5]).all()

    # challenge with top crossed, bottom straight
    assert (paths[2, 0] == [4, 2]).all()
    assert (paths[2, 1] == [2, 3]).all()
    assert (paths[2, 2] == [3, 4]).all()
    assert (paths[2, 3] == [5, 5]).all()

    # challenge with top straight, bottom crossed
    assert (paths[3, 0] == [0, 0]).all()
    assert (paths[3, 1] == [6, 1]).all()
    assert (paths[3, 2] == [7, 6]).all()
    assert (paths[3, 3] == [1, 7]).all()

    assert (paths[4, 0] == [0, 0]).all()
    assert (paths[4, 1] == [6, 1]).all()
    assert (paths[4, 2] == [1, 4]).all()
    assert (paths[4, 3] == [7, 5]).all()

    delay = beli_puf.val(challenges)[:, 0, :]
    assert (delay[0] == [62, 23, 76, 37]).all()
    assert (delay[1] == [0, 41, 14, 55]).all()
    assert (delay[2] == [42, 23, 34, 55]).all()
    assert (delay[3] == [0, 61, 76, 17]).all()
    assert (delay[4] == [0, 61, 14, 75]).all()

    fastest = beli_puf.eval(challenges)
    assert (fastest.squeeze() == [1, 0, 1, 0, 0]).all()

    two_bit_beli_puf = pypuf.simulation.TwoBitBeliPUF(n=4, k=1, seed=1)
    two_bit_beli_puf.delays = beli_puf.delays
    assert (two_bit_beli_puf.eval(challenges)[:, 0] == [1, 1, 1, 1, 1]).all()
    assert (two_bit_beli_puf.eval(challenges)[:, 1] == [-1, 1, -1, 1, 1]).all()

    one_bit_beli_puf = pypuf.simulation.OneBitBeliPUF(n=4, k=1, seed=1)
    one_bit_beli_puf.delays = beli_puf.delays
    assert (one_bit_beli_puf.eval(challenges) == [-1, 1, -1, 1, 1]).all()

    features = beli_puf.features(challenges)
    assert (features[0, 0] == zeros_except(16, [6, 2 + 8])).all()
    assert (features @ beli_puf.delays[0].flatten() == delay).all()


def test_xor_beli_puf():
    n = 32
    for k in [2, 4, 8]:
        for puf_type in [
            pypuf.simulation.TwoBitBeliPUF,
            pypuf.simulation.OneBitBeliPUF,
        ]:
            beli_pufs = [puf_type(n=n, k=1, seed=seed) for seed in range(k)]
            xor_beli_puf = puf_type(n=n, k=k, seed=0)
            for l in range(k):
                xor_beli_puf.delays[l] = beli_pufs[l].delays[0]
            challenges = pypuf.io.random_inputs(n=n, N=2, seed=1)

            responses_ind = np.array([puf.eval(challenges) for puf in beli_pufs]).prod(axis=0)
            responses_xor = xor_beli_puf.eval(challenges)
            assert (responses_ind == responses_xor).all()
