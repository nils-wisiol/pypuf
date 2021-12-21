Beli PUF Attacks
================

[AMTW21]_

Logistic-Regression-Based
-------------------------

>>> import pypuf.simulation, pypuf.io
>>> puf = pypuf.simulation.TwoBitBeliPUF(n=64, k=1, seed=1)
>>> crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=10000, seed=2)



>>> import pypuf.attack
>>> attack = pypuf.attack.TwoBitBeliLR(crps, seed=3, k=1, bs=256, lr=1, epochs=15)
>>> attack.fit()  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
Epoch 1/15
...
39/39 [==============================] ... accuracy: 0.9...
<pypuf.simulation.delay.TwoBitBeliPUF object at 0x...>
>>> model = attack.model

The model accuracy can be measured using the pypuf accuracy metric :meth:`pypuf.metrics.accuracy`.

>>> import pypuf.metrics
>>> pypuf.metrics.similarity(puf, model, seed=4)
array([0.963, 0.958])


API
---

.. autoclass:: pypuf.attack.TwoBitBeliLR
    :members: __init__, fit, beli_output, model, history

.. autoclass:: pypuf.attack.OneBitBeliLR
    :members: __init__, fit, beli_output, model, history

Performance
-----------

===  ====   =====   ============   ===========  =======
  n    k     CRPs   success rate     duration    cores
===  ====   =====   ============   ===========  =======
===  ====   =====   ============   ===========  =======
