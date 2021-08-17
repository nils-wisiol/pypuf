Distance / Similarity
---------------------

In the context of machine learning attacks on PUFs, it is often required to estimate the accuracy of predictions made
by a PUF model. pypuf provides a metric to estimate the similarity of two PUFs, given either as response data or
simulation, which can be used to compute the accuracy of predictions.

.. automodule:: pypuf.metrics
    :members: accuracy, similarity_data, similarity
    :noindex:
