Linear Regression
=================

Linear Regression fits a linear function on given data. The resulting linear function, also called map, is guaranteed
to be optimal with respect to the total squared error, i.e. the sum of the squared differences of actual value and
predicted value.

Linear Regression has many applications, in pypuf, it can be used to model :doc:`../simulation/optical` and
:doc:`../simulation/arbiter_puf`.


Arbiter PUF Reliability Side-Channel Attack [DV13]_
---------------------------------------------------

For Arbiter PUFs, the reliability for any given challenge :math:`c` has a close relationship with the difference in
delay for the top and bottom line. When modeling the Arbiter PUF response as

.. math::
    r = \text{sgn}\left[ D_\text{noise} + \langle w, x \rangle \right],

where :math:`x` is the feature vector corresponding to the challenge :math:`c` and :math:`w \in \mathbb{R}^n` are the
weights describing the Arbiter PUF, and :math:`D_\text{noise}` is chosen from a Gaussian distribution with zero mean
and variance :math:`\sigma_\text{noise}^2` to model the noise, then we can conclude that

.. math::
    \text{E}[r(x)] = \text{erf}\left( \frac{\langle w, x \rangle}{\sqrt{2}\sigma_\text{noise}} \right).

Hence, the delay difference :math:`\langle w, x \rangle` can be approximated based on an approximation of
:math:`\text{E[r(x)]}`, which can be easily obtained by an attacker. It gives

.. math::
    \langle w, x \rangle = \sqrt{2}\sigma_\text{noise} \cdot \text{erf}^{-1} \text{E}[r(x)].

This approximation works well even when :math:`\text{E}[r(x)]` is approximated based on only on few responses, e.g. 3
(see below).

To demonstrate the attack, we initialize an Arbiter PUF simulation with noisiness chosen such that the reliability
will be about 91% *on average*:

>>> import pypuf.simulation, pypuf.io, pypuf.attack, pypuf.metrics
>>> puf = pypuf.simulation.ArbiterPUF(n=64, noisiness=.25, seed=3)
>>> pypuf.metrics.reliability(puf, seed=3).mean()
0.908...

We then create a CRP set using the *average* value of responses to 500 challenges, based on 5 measurements:

>>> challenges = pypuf.io.random_inputs(n=puf.challenge_length, N=500, seed=2)
>>> responses_mean = puf.r_eval(5, challenges).mean(axis=-1)
>>> crps = pypuf.io.ChallengeResponseSet(challenges, responses_mean)

Based on these approximated values ``responses_mean`` of the linear function :math:`\langle w, x \rangle`, we use
linear regression to find a linear mapping with small error to fit the data. Note that we use the ``transform_atf``
function to compute the feature vector :math:`x` from the challenges :math:`c`, as the mapping is linear in :math:`x`
(but not in :math:`c`).

>>> attack = pypuf.attack.LeastSquaresRegression(crps, feature_map=lambda cs: pypuf.simulation.ArbiterPUF.transform_atf(cs, k=1)[:, 0, :])
>>> model = attack.fit()

The linear map ``model`` will predict the delay difference of a given challenge. To obtain the predicted PUF response,
this prediction needs to be thresholded to either -1 or 1:

>>> model.postprocessing = model.postprocessing_threshold

To measure the resulting model accuracy, we use :meth:`pypuf.metrics.similarity`:

>>> pypuf.metrics.similarity(puf, model, seed=4)
array([0.902])


Modeling Attack on Integrated Optical PUFs [RHUWDFJ13]_
-------------------------------------------------------

The behavior of an integrated optical PUF token can be understood as a linear map
:math:`T \in \mathbb{C}^{n \times m}` of the given challenge, where the value of :math:`T` are determined by the given
PUF token, and :math:`n` is number of challenge pixels, and :math:`m` the number of response pixels.
The speckle pattern of the PUF is a measurement of the intensity of its electromagnetic field at the output, hence the
intensity at a given response pixel :math:`r_i` for a given challenge :math:`c` can be written as

.. math::
    r_i = \left| c \cdot T \right|^2.

pypuf ships a basic simulator for the responses of :doc:`../simulation/optical`, on whose data a modeling attack
can be demonstrated. We first initialize a simulation and collect challenge-response pairs:

>>> puf = pypuf.simulation.IntegratedOpticalPUF(n=64, m=25, seed=1)
>>> crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=1000, seed=2)

Then, we fit a linear map on the data contained in ``crps``. Note that the simulation returns *intensity* values rather
than *field* values. We thus need to account for quadratic terms using an appropriate
:meth:`feature map <pypuf.attack.LeastSquaresRegression.feature_map_optical_pufs_reloaded_improved>`.

>>> attack = pypuf.attack.LeastSquaresRegression(crps, feature_map=pypuf.attack.LeastSquaresRegression.feature_map_optical_pufs_reloaded_improved)
>>> model = attack.fit()

The success of the attack can be visually inspected or quantified by the :doc:`/metrics/correlation` of the response
pixels:

>>> crps_test = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=1000, seed=3)
>>> pypuf.metrics.correlation(model, crps_test).mean()
0.69...

Note that the correlation can differ when additionally, post-processing of the responses is performed, e.g. by
thresholding the values such that half the values give -1 and the other half 1:

>>> import numpy as np
>>> threshold = lambda r: np.sign(r - np.quantile(r.flatten(), .5))
>>> pypuf.metrics.correlation(model, crps_test, postprocessing=threshold).mean()
0.41...


API
---

.. autoclass::
    pypuf.attack.LeastSquaresRegression
    :members: __init__, fit, model, feature_map_optical_pufs_reloaded, feature_map_optical_pufs_reloaded_improved
