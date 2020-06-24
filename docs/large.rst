Large-Scale Experiments
=======================

To assess the quality of an attack or Strong PUF design, it is often necessary to run several attacks for different
parameter settings and many instances. Often, attacks will run long or require much memory.

In pypuf, the application of an attack on a design can be defined as an `experiment`, e.g. the experiment of running
the reliability-based XOR Arbiter PUF attack (:class:`pypuf.attacks.delay.GapAttack`, [Bec15]_) on an XOR Arbiter PUF.

Lists of experiments are grouped in `studies`, which collect results and optionally contain an analysis.

pypuf provides the framework to define `studies`, in which
