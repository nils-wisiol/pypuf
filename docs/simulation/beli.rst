Beli PUF and XOR Beli PUF
=========================

Intro to Beli PUF, Motvation, Design, blah [AMTW21]_.

To simulate a basic Beli PUF with two output bits, use

>>> import pypuf.simulation, pypuf.io
>>> puf = pypuf.simulation.TwoBitBeliPUF(n=32, k=1, seed=1)
>>> crps = pypuf.io.random_inputs(n=32, N=3, seed=2)
>>> puf.eval(crps)
array([[-1, -1],
       [ 1,  1],
       [-1, -1]])

Beli PUF can also output the index of the delay line with the fastest signal for a given challenge,

>>> puf = pypuf.simulation.BeliPUF(n=32, k=1, seed=1)
>>> puf.eval(crps)
array([[3],
       [0],
       [3]])

Observe that the two Beli PUF instances above use the same internal delays and differ only in the output format
specification.

:class:`OneBitBeliPUF` is a Beli PUF version which returns the XOR value of the :class:`TwoBitBeliPUF` version shown
above,

>>> puf = pypuf.simulation.OneBitBeliPUF(n=32, k=1, seed=1)
>>> puf.eval(crps)
array([1, 1, 1])

All Beli PUFs shown above can be arranged into an XOR Beli PUF by using the `k` parameter when instantiating:

>>> puf = pypuf.simulation.OneBitBeliPUF(n=32, k=4, seed=1)
>>> puf.eval(crps)
array([-1,  1, -1])

.. note::
    pypuf currently does not implement noisy Beli PUF simulation.
    However, it ships CRP data of a Beli PUF implemented in FPGA.
    TODO add link.


API
---

.. autoclass:: pypuf.simulation.BeliPUF
    :members: __init__, challenge_length, response_length, eval, val, signal_path, features

.. autoclass:: pypuf.simulation.TwoBitBeliPUF

.. autoclass:: pypuf.simulation.OneBitBeliPUF
