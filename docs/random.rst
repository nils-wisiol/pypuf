Randomness and Reproducibility
==============================

pypuf strives to make all results fully reproducible.

One key ingredient to this is to only use seeded random number generators.
To avoid re-using seeds where different random generators are expected, pypuf implements a convenience function to
obtain random generators based on a string description.

.. automodule:: pypuf.random
    :members: prng
