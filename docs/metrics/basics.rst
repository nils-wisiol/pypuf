Basic Metrics for PUFs
======================

.. note::
    Current implementations of metrics in pypuf do not support :math:`\{0,1\}` bit notation. All challenges and
    responses must be given in :math:`\{-1,1\}` notation.


Reliability
-----------

Reliability is a measure for the noise in challenge evaluation of a single PUF instance. Most PUF implementations
have challenge evaluation mechanisms influenced by noise, hence the question of reproducible responses arises.
In the literature, the concept of reliability is also referred to as `intra-distance`, `stability` and
`reproducibility`. pypuf defines the reliability on challenge :math:`x` of a PUF with responses :math:`\mathtt{eval}(x)`
to be

.. math:: \Pr_\mathtt{eval} \left[ \mathtt{eval}(x) = \mathtt{eval}(x) \right],

where the probability is taken over the noise in the evaluation process :math:`\mathtt{eval}`. The general reliability
of a PUF is the average over all challenges,

.. math:: E_x \left[ \Pr_\mathtt{eval} \left[ \mathtt{eval}(x) = \mathtt{eval}(x) \right] \right].

The definition is applied separately for each output bit.

pypuf ships approximations of PUF reliability based on both an instance of :class:`pypuf.simulation.Simulation` and
on response data given. In some PUFs, the reliability depends on the challenge given, i.e. in one instance, some
challenges have hard-to-reproduce responses, while others have very stable response behavior. pypuf hence reports
reliability information seperatly for each challenge. To obtain a general notion of reliability, results can be averaged
along the first axis.

.. automodule:: pypuf.metrics
    :members: reliability, reliability_data
    :noindex:


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


Distance
--------

In the context of machine learning attacks on PUFs, it is often required to estimate the accuracy of predictions made
by a PUF model. pypuf provides a metric to estimate the similarity of two PUFs, given either as response data or
simulation, which can be used to compute the accuracy of predcitions.

.. automodule:: pypuf.metrics
    :members: accuracy, similarity_data, similarity
    :noindex:
