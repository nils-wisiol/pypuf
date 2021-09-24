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
include URL and file size (if known)::

    Fetching CRPs (9.9MiB) from https://zenodo.org/record/5215875/files/MTZAA20_4XOR_64bit_LUT_2239B_attacking_1M.txt.npz?download=1


Hybrid Boolean Network [CCPG21]_
````````````````````````````````
Hybrid Boolean Networks can be used as PUFs as described by Charlot et al. [CCPG21]_. In pypuf, there are CRPs included
for 8 different PUF instances. Each instance is measured on the same set of 1002 challenges, where each challenge is
measured 100 times. The measurement time is chosen as optimal (cf. discussion in the paper).

=================================  ======================  ==============  ======  ================  ===============  =====================
pypuf Object                       PUF Type                Implementation  Amount  Challenge Length  Response Length  Repeated Measurements
=================================  ======================  ==============  ======  ================  ===============  =====================
``pypuf.io.CCPG21.hbn_board_1``    Hybrid Boolean Network  FPGA              1002               256              256                    100
``pypuf.io.CCPG21.hbn_board_2``    Hybrid Boolean Network  FPGA              1002               256              256                    100
``pypuf.io.CCPG21.hbn_board_3``    Hybrid Boolean Network  FPGA              1002               256              256                    100
``pypuf.io.CCPG21.hbn_board_4``    Hybrid Boolean Network  FPGA              1002               256              256                    100
``pypuf.io.CCPG21.hbn_board_5``    Hybrid Boolean Network  FPGA              1002               256              256                    100
``pypuf.io.CCPG21.hbn_board_6``    Hybrid Boolean Network  FPGA              1002               256              256                    100
``pypuf.io.CCPG21.hbn_board_7``    Hybrid Boolean Network  FPGA              1002               256              256                    100
``pypuf.io.CCPG21.hbn_board_8``    Hybrid Boolean Network  FPGA              1002               256              256                    100


64-bit Interpose PUF [AM21]_
````````````````````````````
The Interpose PUF provided consists of one 64-bit Arbiter PUF in the top layer and five 65-bit Arbiter PUFs in the
bottom layer. The challenges of ``pypuf.io.AM21.interpose_puf`` and ``pypuf.io.AM21.arbiter_puf_top`` are identical;
the challenges to the bottom Arbiter PUFs correspond to the original challenges interposed with the responses of the
top Arbiter PUF.

The ``pypuf.io.AM21.interpose_puf`` CRP data is provided for convenience only, it can also be derived from the other
data sets. Of course, also (1,x)-Interpose PUFs can be build for x < 5.

======================================================  ===================  ==============  ======  ================  ===============  =====================
pypuf Object                                            PUF Type             Implementation  Amount  Challenge Length  Response Length  Repeated Measurements
======================================================  ===================  ==============  ======  ================  ===============  =====================
``pypuf.io.AM21.arbiter_puf_top``                       Arbiter PUF          FPGA                1M                64                1                      1
``pypuf.io.AM21.arbiter_puf_bottom_0``                  Arbiter PUF          FPGA                1M                65                1                      1
``pypuf.io.AM21.arbiter_puf_bottom_1``                  Arbiter PUF          FPGA                1M                65                1                      1
``pypuf.io.AM21.arbiter_puf_bottom_2``                  Arbiter PUF          FPGA                1M                65                1                      1
``pypuf.io.AM21.arbiter_puf_bottom_3``                  Arbiter PUF          FPGA                1M                65                1                      1
``pypuf.io.AM21.arbiter_puf_bottom_4``                  Arbiter PUF          FPGA                1M                65                1                      1
``pypuf.io.AM21.interpose_puf``                         (1,5)-Interpose PUF  FPGA                1M                64                1                      1
======================================================  ===================  ==============  ======  ================  ===============  =====================


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
