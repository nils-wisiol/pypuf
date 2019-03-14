Why Attackers Lose: Design and Security Analysis of Arbitrarily Large XOR Arbiter PUFs
==============

Accepted at the Journal of Cryptographic Engineering 26 Feb 2019, https://doi.org/10.1007/s13389-019-00204-8.

pypuf contains two studies that were used in Why Attackers Lose.

Figure 4(a): Minimum Number of Votes Required
--------------

A graph showing how many votes are needed to achieve stability of at least

    Pr[Stab(c) >= 95%] >= 80%
    
for an XOR Arbiter PUF with 32 bits and k ∈ {2, 4, ..., 32}.

![](../../../figures/why_attackers_lose.fig_04_a.png?raw=true "Why Attackers Lose Figure 4(a)")

Data and figure can be generated with

    python3 -m study why_attackers_lose.fig_04_a

Figure 4(b): Histogram Simulated Stability Distribution With Votes
--------------

A histogram showing the probability density of an XOR Majority Vote
Arbiter PUF of size k = 32 and chain length of n = 32, using 51 and 501 
votes to boot stability, respectively.

![](../../../figures/why_attackers_lose.fig_04_b.png?raw=true "Why Attackers Lose Figure 4(b)")

Data and figure can be generated with

    python3 -m study why_attackers_lose.fig_04_b
