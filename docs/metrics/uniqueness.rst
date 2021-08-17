Uniqueness
----------

Uniqueness is a measure for how differently PUF instances of the same class behave.
As security is build on the individual behavior, a high uniqueness is a minimum security requirement for any PUF.
The concept of uniqueness is also often called `inter-distance`.
pypuf can approximate the uniqueness based on given instances of ``Simulation`` or based on response data, see below.

Uniqueness is estimated on a per-challenge basis, as low uniqueness on a small number of challenges can be problematic.
To obtain a general uniqueness measure for each response bit, average results along the first axis.


.. automodule:: pypuf.metrics
    :members: uniqueness, uniqueness_data
