Overview
--------

Generating Uniform Random Challenges
````````````````````````````````````

.. automodule:: pypuf.io
    :members: random_inputs

Storing Challenge-Response Data
```````````````````````````````

pypuf stores challenge-response data in objects of the class :class:`pypuf.io.ChallengeResponseSet`.
These objects contain two numpy arrays with the challenges and responses, respectively and provide a number of
auxiliary functions for convenient management.

To create a instance of :class:`pypuf.io.ChallengeResponseSet` for given challenges and responses, use

>>> import numpy as np
>>> challenges = np.array([[-1, -1, -1, -1], [-1, -1, -1, 1]])
>>> responses = np.array([1, 1])
>>> import pypuf.io
>>> crp = pypuf.io.ChallengeResponseSet(challenges, responses)

To create an instance of :class:`pypuf.io.ChallengeResponseSet` from a :class:`pypuf.simulation.Simulation`, use

>>> import pypuf.simulation
>>> puf = pypuf.simulation.ArbiterPUF(n=64, seed=1)
>>> import pypuf.io
>>> crp = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=1000, seed=2)

pypuf CRP data can be stored on disk and loaded back into pypuf:

>>> crp = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=1000, seed=2)
>>> crp.save('crps.npz')
>>> crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
>>> crp == crp_loaded
True

:class:`pypuf.io.ChallengeResponseSet` can also be sliced to obtain a single challenge-response pair or to get a
subset of CRPs:

>>> crp = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=1000, seed=2)
>>> crp[0]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
(array([-1,...), array([[1.]]))
>>> crp[:10]
<10 CRPs with challenge length 64 and response length 1, each response measured 1 time(s)>

.. autoclass:: pypuf.io.ChallengeResponseSet
    :members: __init__, challenge_length, response_length, save, load
