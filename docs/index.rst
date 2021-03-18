pypuf: Cryptanalysis of Physically Unclonable Functions
=======================================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3901410.svg
   :target: https://doi.org/10.5281/zenodo.3901410

pypuf is a toolbox for simulation, testing, and attacking Physically Unclonable Functions.
Some functionality is only implemented in the
`archived version of pypuf v1 <https://github.com/nils-wisiol/pypuf/tree/v1>`_.

.. toctree::
   :maxdepth: 2
   :caption: Simulation
   :hidden:

   Overview <simulation/overview>
   simulation/arbiter_puf
   simulation/delay
   simulation/bistable
   simulation/base

.. toctree::
   :maxdepth: 2
   :caption: Metrics
   :hidden:

   Basics <metrics/basics>

.. toctree::
   :maxdepth: 2
   :caption: Tools
   :hidden:

   Randomness and Reproducibility <random>
   Large-Scale Experiments <batch>

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

>>> from pypuf.simulation import XORArbiterPUF
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


Citation
========

To refer to pypuf, please use DOI `10.5281/zenodo.3901410`.
pypuf is published `via Zenodo <https://zenodo.org/badge/latestdoi/87066421>`_.
Please cite this work as

    Nils Wisiol, Christoph Gräbnitz, Christopher Mühl, Benjamin Zengin, Tudor Soroceanu, & Niklas Pirnay.
    pypuf: Cryptanalysis of Physically Unclonable Functions (Version v2, March 2021). Zenodo.
    https://doi.org/10.5281/zenodo.3901410

or use the following BibTeX::

    @software{pypuf,
      author       = {Nils Wisiol and
                      Christoph Gräbnitz and
                      Christopher Mühl and
                      Benjamin Zengin and
                      Tudor Soroceanu and
                      Niklas Pirnay},
      title        = {{pypuf: Cryptanalysis of Physically Unclonable
                       Functions}},
      year         = 2021,
      publisher    = {Zenodo},
      version      = {v2},
      doi          = {10.5281/zenodo.3901410},
      url          = {https://doi.org/10.5281/zenodo.3901410}
    }



About this document
===================

To add to our documentation or fix a mistake, please submit a Pull Request
at https://github.com/nils-wisiol/pypuf.
