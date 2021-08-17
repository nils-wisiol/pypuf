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
