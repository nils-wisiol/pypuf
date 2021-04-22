Arbiter PUFs and Compositions
=============================

Simulation of Arbiter PUFs and compositions of (several) Arbiter PUFs, such as in the XOR Arbiter PUF [SD07]_ or
Interpose PUF [NSJM19]_ as well as designs that use `input transformations` (e.g., Lightweight Secure PUF [MKP08]_
are simulated in pypuf using the :doc:`additive delay model <delay>`.

.. note::
    pypuf uses :math:`\{-1,1\}` to represent bit values in Arbiter PUF simulations, for both challenges and responses.

.. note::
    All simulations based on the additive delay model in pypuf use ``numpy.int8`` to represent bit values, i.e. for each
    challenge or response bit, one byte of memory is allocated. While inefficient in terms of required memory, this
    provides faster evaluation performance than storing one logical bit in one bit of memory.

Arbiter PUF
-----------

The Arbiter PUF appeared first in an article proposing a electrical Strong PUF that can be implemented cheaply using
the regular CMOS manufacturing process [GCvDD02]_. It can be easily modeled by several machine learning attacks.

pypuf supports simulation of Arbiter PUFs with and without noise. To instantiate a random Arbiter PUF instance,
the challenge length ``n`` must be chosen and a ``seed`` for reproducible results must be provided.
To evaluate challenges, provide a `list` of challenges to the ``eval`` method.
For example, to evaluate an Arbiter PUF on three randomly chosen challenges:

>>> from pypuf.simulation import ArbiterPUF
>>> puf = ArbiterPUF(n=64, seed=1)
>>> from pypuf.io import random_inputs
>>> puf.eval(random_inputs(n=64, N=3, seed=2))
array([-1,  1, -1], dtype=int8)

For noisy simulations, a noise level needs to be given. Basically, the higher the number, the more likely responses
are to be disturbed by noise; however the influence of noise also depends on the challenge given and the concrete
instance. For details, please refer to the :doc:`noise model used in the additive delay model <delay>`.

The noise level is specified as the `noisiness` parameter, ``.1`` is a realistic value; ``1.0`` will result in purely
random responses. ``0.0`` will disable noise simulation, which is the default.

>>> from pypuf.simulation import ArbiterPUF
>>> puf = ArbiterPUF(n=64, seed=1, noisiness=.2)
>>> from pypuf.io import random_inputs
>>> puf.eval(random_inputs(n=64, N=3, seed=2))
array([-1, -1,  1], dtype=int8)
>>> puf.eval(random_inputs(n=64, N=3, seed=2))
array([-1, -1, -1], dtype=int8)

In above example, note that the response to the `same` third challenge evaluates to a different response due to noise.
This is reproducible with new instantiations using the same seed.

XOR Arbiter PUF
---------------

The XOR Arbiter PUF was introduced as mitigation for the vulnerability of the Arbiter PUF to machine learning attacks
[SD07]_. It consists of ``k`` Arbiter PUFs with all receive the same input, the response is defined as the parity (XOR)
of the individual responses. The security of XOR Arbiter PUF was first studied by Rührmair et al. [RSSD10]_, who
successfully attacked XOR Arbiter PUFs using the responses of random challenges. Their attack scales polynomially in
the challenge length and exponentially in the number of parallel arbiter chains. High-performance results of this attack
were studied by Tobisch and Becker [TB15]_. The must successful attack on XOR Arbiter PUFs is based on the noise level
of challenges, as measured by the attacker [Bec15]_.

To simulate an 8-XOR 64-bit XOR Arbiter PUF using relatively stable Arbiter PUF implementations, use

>>> from pypuf.simulation import XORArbiterPUF
>>> puf = XORArbiterPUF(n=64, k=8, seed=1, noisiness=.05)
>>> from pypuf.io import random_inputs
>>> puf.eval(random_inputs(n=64, N=3, seed=2))
array([-1,  1, -1], dtype=int8)

.. note::
    The `noisiness` parameter in the XOR Arbiter PUF is directly referring to the noise of `each` Arbiter PUF in the
    simulation. This means that for equal `noisiness`, noise level will increase for higher `k`.

Feed-Forward (XOR) Arbiter PUF
------------------------------

Feed-Forward Arbiter PUFs are an attempt to increase resistance to modeling attacks [GLCDD04]_ compared to traditional
Arbiter PUFs. In an Feed-Forward XOR Arbiter PUF, many Feed-Forward Arbiter PUFs are evaluated in parallel, and the
XOR of the individual response bits is returned. The feed-forward loops may be homogeneous, i.e. the same for all
participating Arbiter PUFs, or inhomogeneous.

Even if using the noisy simulation, all involved arbiter elements are assumed to work entirely noise-free with infinite
precision.

To simulate an 4-XOR 128-bit Feed Forward Arbiter PUF in which each Feed-Forward Arbiter PUF will have a feed-forward
loop after the 33nd stage that will determine the challenge bit to the 69th stage, use

>>> from pypuf.simulation import XORFeedForwardArbiterPUF
>>> puf = XORFeedForwardArbiterPUF(n=128, k=4, ff=[(32, 68)], seed=1)
>>> from pypuf.io import random_inputs
>>> puf.eval(random_inputs(n=128, N=6, seed=2))
array([-1, -1,  1,  1, -1,  1])

The full Feed-Forward Arbiter PUF simulation API is displayed below.

.. automethod:: pypuf.simulation.FeedForwardArbiterPUF.__init__
.. automethod:: pypuf.simulation.XORFeedForwardArbiterPUF.__init__


Lightweight Secure PUF
----------------------

The Lightweight Secure PUF [MKP08]_ was introduced to mitigate the vulnerability of the Arbiter PUF to machine learning
attacks, and is the first PUF that uses `different` challenges to each arbiter chain, all generated from a `master
challenge`. The Lightweight Secure PUF makes naive attacks harder [RSSD10]_, however does not increase overall attack
resilience [WBMS19]_.

.. todo::
    Add information on how the input transformation is defined and implemented.

To simulate an 8-XOR 64-bit XOR Arbiter PUF using relatively stable Arbiter PUF implementations, use

>>> from pypuf.simulation import LightweightSecurePUF
>>> puf = LightweightSecurePUF(n=64, k=8, seed=1, noisiness=.05)
>>> from pypuf.io import random_inputs
>>> puf.eval(random_inputs(n=64, N=3, seed=2))
array([ 1, -1, -1], dtype=int8)


Permutation PUF
---------------

The Permutation PUF is an iteration of the idea behind the Lightweight Secure PUF, which is to feed different
challenges to the arbiter chains in an XOR Arbiter PUF. After it was shown that the specific way the Lightweight Secure
PUF modifies the individual challenges, the Permutation PUF was introduced to simplify implementation and remove the
attack surface the Lightweight Secure PUF introduced [WBMS19]_.

To generate the individual challenges, the Permutation PUF applies a predetermined set of `k` permutations of the
`master` challenge, one for each individual challenge. The permutations are chosen in a way such that no two
permutations permute a bit the same way, i.e. from the same position to the same position, and additionally such that
no permutation has a fix point.

To simulate an 8-XOR 64-bit XOR Arbiter PUF using relatively stable Arbiter PUF implementations, use

>>> from pypuf.simulation import PermutationPUF
>>> puf = PermutationPUF(n=64, k=8, seed=1, noisiness=.05)
>>> from pypuf.io import random_inputs
>>> puf.eval(random_inputs(n=64, N=3, seed=2))
array([-1, -1, -1], dtype=int8)


Interpose PUF
-------------

The Interpose PUF [NSJM19]_ was designed to mitigate the well-performing reliability-based attack on the XOR Arbiter
PUF [Bec15]_. It consists of two XOR Arbiter PUFs, called `upper` and `lower` layer. The upper layer has
:math:`k_\text{up}` parallel arbiter chains and challenge length :math:`n`, the lower layer :math:`k_\text{down}` and
challenge lenght :math:`n+1`. To determine the response of the Interpose PUF, the challenge is input into the upper
layer and evaluated. The response of the upper layer is then `interposed` in the middle of the challenge; the resulting
:math:`n+1` bit long challenge is then input in the lower layer. The resulting response is the final response of the
PUF.

A security analysis of showed the Interpose PUF to be immune against known attacks in the literature [NSJM19]_.
However, the Logistic Regression attack [RSSD10]_, originally designed for attacking the XOR Arbiter PUF, can be
modified to "split" the Interpose PUF and model it with effort only slightly above what is needed to attack XOR
Arbiter PUFs of similar size [WMPN19]_.

To simulate an (8,8) 64-bit Interpose PUF using relatively stable Arbiter PUF implementations, use

>>> from pypuf.simulation import InterposePUF
>>> puf = InterposePUF(n=64, k_up=8, k_down=8, seed=1, noisiness=.05)
>>> from pypuf.io import random_inputs
>>> puf.eval(random_inputs(n=64, N=3, seed=2))
array([ 1, -1, -1], dtype=int8)

Note that the ``noisiness`` parameter applies to both upper and lower layer.
