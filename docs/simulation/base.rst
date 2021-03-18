Base Class for Simulations
==========================

Batch Evaluation
----------------

pypuf PUF simulations are build to evaluate several given challenges at once, to provide better performance. The
interfaces are defined accordingly, see below.


Adding Simulations
------------------

Simulations should extend the base class ``simulation.Simulation`` and provide a instantiation function that
takes a ``seed`` argument for reproducible results. Weak PUFs can be added by setting ``challenge_length`` to zero.


API
---

.. autoclass:: pypuf.simulation.Simulation
    :member-order: bysource
    :members:
