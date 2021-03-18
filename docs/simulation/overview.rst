PUF Simulations in pypuf
========================

Modelling of PUFs plays a central role in their analysis.
Without a theoretical model, we can make no predictions on the behavior of PUFs and hence cannot study its security.
On the other hand, often a model can be used to simulate PUF behavior to facilitate the analysis and avoid expensive
studies with hardware.

Included PUF Desings
--------------------
pypuf currently features simulation of the following strong PUFs:

1. :doc:`Arbiter PUF-based designs <arbiter_puf>` utilizing the
   :doc:`additive delay model <delay>`, including simulations of the Arbiter PUF [GCvDD02]_, XOR Arbiter PUF [SD07]_,
   Lightweight Secure PUF [MKP08]_, Permutation PUF [WBMS19]_, and Interpose PUF [NSJM19]_.
2. :doc:`Feed-Forward Arbiter PUFs <arbiter_puf>` and XORs thereof [GLCDD04]_.
3. :doc:`PUF designs based on bistable rings <bistable>` [CCLSR11]_ [XRHB15]_.

Technicalities
--------------
Simulations of PUFs in pypuf are build for performance and hence :doc:`use batch evaluation <base>` of challenges:
given a list of challenges, a list of responses will be returned.
