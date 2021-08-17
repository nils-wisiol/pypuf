LMN Algorithm
=============

The LMN Algorithm can compute models for functions with Fourier spectra concentrated on the low degrees [LMN93]_,
[ODon14]_ and part of the PUFMeter [GFS19]_ PUF testing toolbox.

The attack requires access to a number of uniformly random challenge-response pairs (CRPs). Depending on the amount of
provided CRPs and the type of function provided, the results may have, with certain probability, a guaranteed accuracy.
(For more details, we refer the reader to the PAC learning framework [ODon14]_.)

Example Usage
-------------

To run the attack, CRP data of the PUF token under attack is required. Such data can be obtained through experiments
on real hardware, or using a simulation. In this example, we use the pypuf Arbiter PUF simulator and configure it to
use feature vectors as inputs rather than challenge vectors by setting ``transform='id'``:

>>> import pypuf.simulation, pypuf.io
>>> puf = pypuf.simulation.ArbiterPUF(n=32, transform="id", seed=1)
>>> challenges = pypuf.io.random_inputs(n=32, N=2000, seed=2)
>>> crps = pypuf.io.ChallengeResponseSet(challenges, puf.val(challenges))

To run the attack, we need to decide how many levels of Fourier coefficients we want to approximate. There are
:math:`n` levels, the :math:`i`-th level has :math:`\binom{n}{i}` coefficients. The run time of the attack is linear
in the number of coefficients. With the steeply increasing run time, in practice, degree 1 and degree 2 are reasonable
choices. Increasing the degree further will not only lead to high requirements on computation time, but may actually
`worsen` the predictive accuracy, as more coefficients need to be approximated.

>>> import pypuf.attack
>>> attack = pypuf.attack.LMNAttack(crps, deg=1)
>>> model = attack.fit()

The model accuracy can be measured using the pypuf accuracy metric :meth:`pypuf.metrics.accuracy`.

>>> import pypuf.metrics
>>> pypuf.metrics.similarity(puf, model, seed=4)
array([0.95])

API
---

.. autoclass:: pypuf.attack.LMNAttack
    :members: __init__, fit, model
