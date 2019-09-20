XOR Arbiter PUFs have Systematic Response Bias 
==============

pypuf contains two studies that were used in XOR Arbiter PUFs have Systematic Response Bias.


Figure 2: Response Bias of XOR Arbiter PUFs, Lightweight Secure PUFs and Permutation Arbiter PUFs
--------------

Analysis results for simulated 64-bit k-XOR Arbiter PUFs, k-Lightweight Secure PUFs and k-Permutation XOR Arbiter PUFs 
build from unbiased Arbiter PUFs. For each type and size, 5000 instances were sampled and queried with one million 
uniformly random challenges each.

Histogram of bias estimates. A bias value of zero represents perfectly unbiased responses.

![](../../../figures/bias.xor_distribution.png?raw=true "Response Bias Distribution")

Proportion of instances that passed the NIST frequency test at significance level 1%.

![](../../../figures/bias.xor_distribution.test_scores.png?raw=true "NIST Frequency Test Pass Proportion")

Data and figure can be generated with

    python3 -m study bias.xor_distribution


Figure 3: Response Bias of Interpose PUFs
--------------

Analysis results for simulated 64 bit (k_up, k_down)-iPUF instances build from unbiased Arbiter PUFs. For each size, 
5000 instances were sampled and queried with one million uniformly random challenges each.

Histogram of bias estimates. A bias value of zero represents perfectly unbiased responses.

![](../../../figures/bias.ipuf_distribution.png?raw=true "Response Bias Distribution")

Proportion of instances that passed the NIST frequency test at significance level 1%.

![](../../../figures/bias.ipuf_distribution.test_scores.png?raw=true "NIST Frequency Test Pass Proportion")

Data and figure can be generated with

    python3 -m study bias.ipuf_distribution
