PUF Data Sets
-------------

pypuf includes a number of data sets which have been published in the scientific literature on PUFs.
The data is not downloaded along with pypuf when it is installed, instead, it is automatically fetched on first
access.

For example, pypuf includes an FPGA-implemented 64-bit 4-XOR Arbiter PUF [MTZAA20]_:

>>> import pypuf.io
>>> pypuf.io.MTZAA20.xor_arbiter_puf_4_xor
<CRP Set available from https://zenodo.org/record/5215875/files/MTZAA20_4XOR_64bit_LUT_2239B_attacking_1M.txt.npz?download=1, not fetched yet>

The data is automatically fetched on first use, e.g. to compute the bias of the responses:

>>> import pypuf.metrics
>>> pypuf.metrics.bias_data(pypuf.io.MTZAA20.xor_arbiter_puf_4_xor.responses)
array([[0.03495]])

As CRP sets may have large size and significnat download time, pypuf logs a warning message when fetching CRPs, which
include URL and file size (if known):

.. code-block::

    Fetching CRPs (9.9MiB) from https://zenodo.org/record/5215875/files/MTZAA20_4XOR_64bit_LUT_2239B_attacking_1M.txt.npz?download=1


Included Data Sets
~~~~~~~~~~~~~~~~~~

64-bit XOR Arbiter PUF [MTZAA20]_
`````````````````````````````````

======================================================  =================  ==============  ======  ================  ===============  =====================
pypuf Object                                            PUF Type           Implementation  Amount  Challenge Length  Response Length  Repeated Measurements
======================================================  =================  ==============  ======  ================  ===============  =====================
``pypuf.io.MTZAA20.xor_arbiter_puf_4_xor``              4-XOR Arbiter PUF  FPGA                1M                64                1                      1
``pypuf.io.MTZAA20.xor_arbiter_puf_5_xor``              5-XOR Arbiter PUF  FPGA                1M                64                1                      1
``pypuf.io.MTZAA20.xor_arbiter_puf_6_xor``              6-XOR Arbiter PUF  FPGA                1M                64                1                      1
``pypuf.io.MTZAA20.xor_arbiter_puf_7_xor``              7-XOR Arbiter PUF  FPGA                5M                64                1                      1
``pypuf.io.MTZAA20.xor_arbiter_puf_8_xor``              8-XOR Arbiter PUF  FPGA                5M                64                1                      1
``pypuf.io.MTZAA20.xor_arbiter_puf_9_xor``              9-XOR Arbiter PUF  FPGA                5M                64                1                      1
======================================================  =================  ==============  ======  ================  ===============  =====================
