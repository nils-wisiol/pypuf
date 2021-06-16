Multilayer Perceptron Attack
============================

Multilayer Perceptron (MLP) was first used by Alkatheiri and Zhuang [AZ17]_ to model Feed-Forward Arbiter PUFs.
In follow-up work, Aseeri et al. [AZA18]_ launched MLP-based modeling attacks against XOR Arbiter PUFs. Thereafter,
Mursi et al. [MTZAA20]_ and Wisiol et al. [WMSZ21]_ modified the network and parameters used by Aseeri et al. to reduce
data complexity of the attack.

pypuf contains two closely related MLP-based modeling attacks. The state-of-the-art attack by Wisiol et al. [WMSZ21]_
and a re-implementation of the attack by Aseeri et al. [AZA18]_ using tensorflow/Keras.

Example Usage [WMSZ21]_
-----------------------

To run the attack, CRP data of the PUF token under attack is required. Such data can be obtained through experiments
on real hardware, or using a simulation. In this example, we use the pypuf XOR Arbiter PUF simulator:

>>> import pypuf.simulation, pypuf.io
>>> puf = pypuf.simulation.XORArbiterPUF(n=64, k=5, seed=1)
>>> crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=500000, seed=2)

To run the attack, we configure the attack object with the challenge response data and attack parameters. The parameters
need careful adjustment for each choice of security parameters in the PUF. Then the attack is run using the
:meth:`pypuf.attack.MLPAttack2021.fit` method.

>>> import pypuf.attack
>>> attack = pypuf.attack.MLPAttack2021(
...     crps, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4],
...     epochs=30, lr=.001, bs=1000, early_stop=.08
... )
>>> attack.fit()  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    Epoch 1/30
    ...
    495/495 [==============================] - ... - loss: 0.0... - accuracy: 0.9... - val_loss: 0.0670 - val_accuracy: 0.9750
    <pypuf.attack.mlp2021.MLPAttack2021.Model object at 0x...>
>>> model = attack.model

The model accuracy can be measured using the pypuf accuracy metric :meth:`pypuf.metrics.accuracy`.

>>> import pypuf.metrics
>>> pypuf.metrics.similarity(puf, model, seed=4)
array([0.97])

Example Usage [AZA18]_
----------------------

The implementation of the modeling attack by Aseeri et al. [AZA18]_ in pypuf is very similar to the version by
Wisiol et al. [WMSZ21]_ given above, with notable differences in the parameter settings given to the attack, in
the memory management, and framework used. While the attack by Aseeri et al. uses scikit learn, pypuf's implementation
is Keras-based. To run the original attack using pypuf, use the network size as defined by Aseeri et al., i.e.
:math:`(2^k, 2^k, 2^k)`, and set the activation function of the hidden layers to ReLU. pypuf does not support the
memory management introduced by Aseeri et al.

>>> puf = pypuf.simulation.XORArbiterPUF(n=64, k=5, seed=1)
>>> crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=800000, seed=2)
>>> attack = pypuf.attack.MLPAttack2021(
...     crps, seed=3, net=[2 ** 5, 2 ** 5, 2 ** 5],
...     epochs=30, lr=.001, bs=1000, early_stop=.08,
...     activation_hl='relu',
... )
>>> model = attack.fit()  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    Epoch 1/30
    ...
>>> pypuf.metrics.similarity(puf, model, seed=4)[0] > .9
True

Note that this is only an approximation of the original work of Aseeri et al., further differences may exist.

Applicability [WMSZ21]_
-----------------------

The attack is noise-resilient and successfully models XOR Arbiter PUFs even if the available training data has response
(label) noise [WMSZ21]_.

This implementation is also suitable to conduct the splitting attack [WMPN19]_ on the Interpose PUF [WMSZ21]_.

API
---

.. autoclass:: pypuf.attack.MLPAttack2021
    :members: __init__, fit, model, history


Performance [WMSZ21]_
---------------------

The pypuf implementation is tested using tensorflow 2.4 on Intel Xeon E5-2630 v4 attacking :math:`n`-bit :math:`k`-XOR
Arbiter PUFs. Memory recorded here may be higher than actually needed by the most recent version of the attack.

===========  ===  ====   =====   ============   ===========  =======  =======
reliability   n    k     CRPs    success rate    duration     cores   memory
===========  ===  ====   =====   ============   ===========  =======  =======
       1.00   64    4     150k          10/10      <1 min       40      1 GiB
       1.00   64    5     200k          10/10      <1 min       20      3 GiB
       1.00   64    6       2M          10/10      <1 min       40      2 GiB
       1.00   64    7       4M          10/10      <1 min       40      3 GiB
       1.00   64    8       6M           7/10      13 min        4
       1.00   64    9      45M          10/10      16 min       40     14 GiB
       1.00   64   10     119M           7/10     291 min       40     41 GiB
       1.00   64   11     325M          10/10    1898 min       40    104 GiB

       1.00  128    4       1M            9/9      <1 min       40      1 GiB
       1.00  128    5       1M          10/10      <1 min       40      2 GiB
       1.00  128    6      10M           9/10      <1 min       20      5 GiB
       1.00  128    7      30M          10/10       2 min       20     20 GiB

       1.00  256    4       6M          10/10       1 min       40      6 GiB
       1.00  256    5      10M          10/10       3 min       40     11 GiB
       1.00  256    6      30M            0/8          --       40     33 GiB
       1.00  256    7     100M           1/10       8 min       40     98 GiB

       0.85   64    4     180k           9/10       <1 min       4     <1 GiB
       0.85   64    5     150k          10/10       <1 min       4     <1 GiB
       0.85   64    6       2M          10/10       <1 min       4      1 GiB
       0.85   64    7       4M            9/9        3 min       4      2 GiB
===========  ===  ====   =====   ============   ===========  =======  =======
