Logistic Regression Attack
==========================

The Logistic Regression Attack (LR Attack) was introduced by Rührmair et al. [RSSD10]_ to model XOR Arbiter PUFs.
In pypuf, an modernized and optimized version of the LR attack is implemented using tensorflow. The
`original implementation <http://www.pcp.in.tum.de/code/lr.zip>`_ is also available from the authors. A study of the
data complexity of the LR attack using a different unpublished implementation of the LR attack was done by Tobisch and
Becker [TB15]_.

Example Usage
-------------

To run the attack, CRP data of the PUF token under attack is required. Such data can be obtained through experiments
on real hardware, or using a simulation. In this example, we use the pypuf XOR Arbiter PUF simulator:

>>> import pypuf.simulation, pypuf.io
>>> puf = pypuf.simulation.XORArbiterPUF(n=64, k=4, seed=1)
>>> crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=50000, seed=2)

To run the attack, we configure the attack object with the challenge response data and attack parameters. The parameters
need careful adjustment for each choice of security parameters in the PUF. Then the attack is run using the
:meth:`pypuf.attack.LRAttack2021.fit` method.

>>> import pypuf.attack
>>> attack = pypuf.attack.LRAttack2021(crps, seed=3, k=4, bs=1000, lr=.001, epochs=100)
>>> attack.fit()  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    Epoch 1/100
    ...
    50/50 [==============================] - ... - loss: 0.4... - accuracy: 0.9... - val_loss: 0.4643 - val_accuracy: 0.9620
    <pypuf.simulation.base.LTFArray object at 0x...>
>>> model = attack.model

The model accuracy can be measured using the pypuf accuracy metric :meth:`pypuf.metrics.accuracy`.

>>> import pypuf.metrics
>>> pypuf.metrics.similarity(puf, model, seed=4)
array([0.966])

Applicability
-------------

The Logistic Regression can be extended to other variants of the XOR Arbiter PUF [WBMS19]_. This functionality is
currently not yet included in pypuf, so that the user needs to perform the appropriate conversions before using
:class:`pypuf.attack.LRAttack2021`.

This implementation is also suitable to conduct the splitting attack on the Interpose PUF [WMPN19]_.

API
---

.. autoclass:: pypuf.attack.LRAttack2021
    :members: __init__, fit, model, history


Performance
-----------

The pypuf implementation is tested using tensorflow 2.4 on Intel Xeon E5-2630 v4 attacking :math:`n`-bit :math:`k`-XOR
Arbiter PUFs. The results below are compared with those of Tobisch and Becker [TB15]_, which have been obtained in 2015
using up to 16 cores.

===  ====   =====   ============   ===========  =======  ==================
  n    k     CRPs   success rate     duration    cores   [TB15]_ / 16 cores
===  ====   =====   ============   ===========  =======  ==================
 64    4      30k          10/10       <1 min       4           <1 min
 64    5     260k          10/10        4 min       4           <1 min
 64    6       2M          20/20       <1 min       4            1 min
 64    7      20M          10/10        3 min       4           55 min
 64    8     150M          10/10       28 min       4          391 min
 64    9     350M                                             2266 min
 64    9     500M           7/10       14 min      40
 64   10       1B           6/10       41 min      40
===  ====   =====   ============   ===========  =======  ==================
