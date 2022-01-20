import pytest

import pypuf.io
import pypuf.metrics
from pypuf.simulation import OneBitBeliPUF, TwoBitBeliPUF
from pypuf.attack import OneBitBeliLR, TwoBitBeliLR


def run_attack(puf_class, k, attack_class,
               n=32, N=20000, bs=256, lr=1, seed=1, epochs=5, stop_validation_accuracy=.95, verbose=False):
    puf = puf_class(n=n, k=k, seed=seed)
    crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=N + 1000, seed=seed + 1)
    attack = attack_class(
        crps, seed=seed + 2,
        k=k, bs=bs, lr=lr, epochs=epochs,
        stop_validation_accuracy=stop_validation_accuracy,
    )
    attack.fit(verbose=verbose)
    return pypuf.metrics.similarity(puf, attack.model, seed=seed + 3).min()


def test_two_bit_beli_puf_attack():
    assert run_attack(puf_class=TwoBitBeliPUF, k=1, attack_class=TwoBitBeliLR) > .9


def test_one_bit_beli_puf_attack():
    assert run_attack(puf_class=OneBitBeliPUF, k=1, attack_class=OneBitBeliLR) > .9


def test_xor_two_bit_beli_puf_attack():
    assert run_attack(puf_class=TwoBitBeliPUF, k=2, attack_class=TwoBitBeliLR) > .9


def test_xor_one_bit_beli_puf_attack():
    assert run_attack(puf_class=OneBitBeliPUF, k=2, attack_class=OneBitBeliLR) > .9


@pytest.mark.parametrize("puf_class, attack_class", [(OneBitBeliPUF, OneBitBeliLR), (TwoBitBeliPUF, TwoBitBeliLR)])
@pytest.mark.parametrize("k, N", [(2, 100000), (3, 100000), (4, 150000)])
def test_large_xor_attack(puf_class, attack_class, k, N):
    assert run_attack(puf_class=puf_class, attack_class=attack_class, k=k, n=32, N=N) > .9
