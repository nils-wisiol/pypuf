Additive Delay Model
====================

The Additive Delay Model [GCvDD02]_ can be used to model Arbiter PUFs as well as Arbiter PUF-based constructions,
such as the XOR Arbiter PUF, the Lightweight Secure PUF, the Permutation PUF, and the Interpose PUF.
It can be extended by a noise model based on Gaussian random variables, which has been shown to highly accurate [DV13]_.

This module contains ``LTFArray``, a simulation implementation of the Additive Delay Model, as well as
``NoisyLTFArray``, an implementation of the Additive Delay Model with Gaussian noise.

.. note::
    pypuf's implementation of the Additive Delay Model uses :math:`\{-1,1\}` to represent bit values both for
    challenges and responses.

.. note::
    All simulations based on the Additive Delay Model in pypuf use ``numpy.int8`` to represent bit values, i.e. for each
    challenge or response bit, one byte of memory is allocated. While inefficient in terms of required memory, this
    provides faster evaluation performance than storing one logical bit in one bit of memory.


Modeling of Delay Values
------------------------

.. todo:: Add derivation of Additive Delay Model from circuit

Implementation of the Additive Delay Model

.. autoclass:: pypuf.simulation.base.LTFArray
    :members: eval


Modeling of Noise
-----------------

.. todo:: Add overview of ``NoisyLTFArray``.
