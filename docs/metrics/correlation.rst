Correlation
-----------

The correlation metric is useful to judge the prediction accuracy when responses are non-binary, e.g. when studying
:doc:`/simulation/optical`.

>>> import pypuf.simulation, pypuf.io, pypuf.attack, pypuf.metrics
>>> puf = pypuf.simulation.IntegratedOpticalPUF(n=64, m=25, seed=1)
>>> crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=1000, seed=2)
>>> crps_test = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=1000, seed=3)
>>> feature_map = pypuf.attack.LeastSquaresRegression.feature_map_optical_pufs_reloaded_improved
>>> model = pypuf.attack.LeastSquaresRegression(crps, feature_map=feature_map).fit()
>>> pypuf.metrics.correlation(model, crps_test).mean()
0.69...

Note that the correlation can differ when additionally, post-processing of the responses is performed, e.g. by
thresholding the values such that half the values give -1 and the other half 1:

>>> import numpy as np
>>> threshold = lambda r: np.sign(r - np.quantile(r.flatten(), .5))
>>> pypuf.metrics.correlation(model, crps_test, postprocessing=threshold).mean()
0.41...

.. automodule:: pypuf.metrics
    :members: correlation, correlation_data
    :noindex:
