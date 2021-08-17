Bias
----

The response bias is one of the fundamental metrics for the response behavior of PUFs. Large response bias enable
trivial modeling attacks on the PUF as the attacker can just guess the more likely of the two responses.

PUF bias is in the literature also known as *balance* or *uniformity*.

Note that per bit encoding in pypuf, a bias of *zero* means a perfectly unbiased behavior, whereas bias of -1 and 1
means that the PUF under test always returns -1 and 1, respectively. To convert the pypuf bias value :math:`b` into the
more traditional 0-1-scale, use :math:`1/2 - b/2`.

.. automodule:: pypuf.metrics
    :members: bias, bias_data
    :noindex:
