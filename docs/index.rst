pypuf: Cryptanalysis of Physically Unclonable Functions
=======================================================

pypuf is a toolbox for simulation, testing, and attacking Physically Unclonable Functions.

.. warning::
    This cleaned up version does not yet contain PUF metrics and attacks. Project structure and APIs are still subject
    to change.

.. toctree::
   :maxdepth: 2
   :caption: Simulation
   :hidden:

   Overview <simulation/overview>
   simulation/arbiter_puf
   simulation/delay
   simulation/base

.. toctree::
  :maxdepth: 2
  :caption: Appendix
  :hidden:

  appendix/acknowledgements
  appendix/bibliography

Getting Started
===============

pypuf is available via pypi::

    pip3 install pypuf

To simulate an XOR Arbiter PUF with 64 bit challenge length and 2 arbiter chains, follow these steps.

First, create a new XOR Arbiter PUF instance with the given dimensions and a fixed seed for reproducibility.

>>> from pypuf.simulation.delay import XORArbiterPUF
>>> puf = XORArbiterPUF(n=64, k=2, seed=1)

Then generate a list of `N` random challenges of length 64 (again with a seed for reproducibility).

>>> from pypuf.io import random_inputs
>>> challenges = random_inputs(n=64, N=10, seed=2)

Finally, evaluate the XOR Arbiter PUF on these challenges, it will yield 10 responses.

>>> puf.eval(challenges)
array([-1, -1, -1,  1,  1, -1,  1, -1,  1, -1], dtype=int8)

For a more detailed information on simulation of PUFs, continue with
:doc:`the section on simulations <simulation/overview>`.

Getting Help
============

If you need help beyond this documentation, please contact me at pypuf(a-t)nils-wisiol.de.


About this document
===================

To add to our documentation or fix a mistake, please submit a Pull Request
at https://github.com/nils-wisiol/pypuf.
