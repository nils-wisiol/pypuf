Bistable Ring PUFs and Compositions
===================================

Simulation of all flavors of Bistable Ring PUFs in pypuf are based on a model based in linear threshold functions.
This model was found to describe Bistable Rings and Twisted Bistable Rings quite well [XRHB15]_. Notice that the
Bistable Ring PUF simulation in pypuf is based on a model derived from successful modeling attacks [XRHB15]_ rather
than a physically motivated model.

.. warning::
    pypuf is currently unaware of how physical intrinsics of Bistable Rings are generated in the PUF manufacturing
    process, hence passing of predetermined weights is mandatory for instantiating any bistable ring PUF.

.. note::
    pypuf uses :math:`\{-1,1\}` to represent bit values in Bistable Ring PUF simulations, for both challenges and
    responses.

.. note::
    All bistable ring PUF simulations in pypuf use ``numpy.int8`` to represent bit values, i.e. for each
    challenge or response bit, one byte of memory is allocated. While inefficient in terms of required memory, this
    provides faster evaluation performance than storing one logical bit in one bit of memory.

Bistable Ring PUF
-----------------

The Bistable Ring PUF was proposed as an FPGA-friendly Strong PUF design [CCLSR11]_.

pypuf supports simulation of Bistable Rings only without noise, due to the lack of a noise model in the literature.
To instantiate a Bistable Ring PUF instance, the challenge length ``n`` must be chosen and appropriate weight parameters
must be provided.
To evaluate challenges, provide a `list` of challenges to the ``eval`` method.
For example, to obtain a couple of Challenge-Response-Pairs of an Bistable Ring PUF:

>>> from numpy.random import default_rng
>>> from pypuf.simulation import BistableRingPUF
>>> n = 64
>>> weights = default_rng(1).normal(size=(n+1))  # instead, data should be derived from experimental setup
>>> puf = BistableRingPUF(n=64, weights=weights)
>>> from pypuf.io import ChallengeResponseSet
>>> crps = ChallengeResponseSet.from_simulation(puf, N=3, seed=1)
>>> crps.responses[:, 0, 0]  # first response bit, first query
array([ 1., -1., -1.])


XOR Bistable Ring PUF
---------------------

After successful modeling attacks on the Bistable Ring PUF [SH14]_, an XORed version of the Bistable Ring PUF was
suggested [XRHB15]_. Much like the XOR Arbiter PUF, it is a design variant where many Bistable Ring PUFs are
instantiated and the parity of the individual responses is given as the final output of the XOR Bistable Ring PUF.

To simulate an 8-XOR 64-bit Bistable Ring PUF, use

>>> from numpy.random import default_rng
>>> from pypuf.simulation import XORBistableRingPUF
>>> k, n = 8, 64
>>> weights = default_rng(1).normal(size=(k, n+1))  # instead, data should be derived from experimental setup
>>> puf = XORBistableRingPUF(n=64, k=8, weights=weights)
>>> from pypuf.io import random_inputs
>>> puf.eval(random_inputs(n=64, N=4, seed=2))
array([ 1,  1,  1, -1], dtype=int8)
