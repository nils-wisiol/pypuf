#  Splitting the Interpose PUF: A Novel Modeling Attack Strategy

pypuf contains two studies that were used in "[Splitting the Interpose PUF](https://eprint.iacr.org/2019/1473)".


## Divide-and-Conquer Attack on Interpose PUFs: `ipuf.split`

Conducts the Divide-and-Conquer attacks outlined in the paper on a large set of Interpose PUFs for different parameters.
Results are included in this repository (`results/ipuf.split.csv.gz`).
The study generates a table and three figures that are also shown in the paper.

### Attack Run Times and CRPs Required for "Popular" Interpose PUF Parameterizations

iPUF size |  best #CRP |  reliability | memory used(*1) |      t1 (#threads/CPU) |  success rate |   samples 
--------- | ---------- | ------------ | --------------- | ---------------------- | ------------- | ---------
  (1, 5)  |       500k |          0.8 |             4.9 |     10.36min (1/CPU-2) |          1.00 |       100 
  (1, 5)  |       500k |          0.9 |             4.4 |      8.70min (1/CPU-2) |          1.00 |       100 
  (1, 5)  |       500k |          1.0 |             4.2 |      9.14min (1/CPU-2) |          1.00 |       100 
  (1, 6)  |         2M |          0.8 |            14.5 |        1.62h (1/CPU-2) |          1.00 |        57 
  (1, 6)  |         2M |          1.0 |            12.2 |        1.48h (1/CPU-2) |          1.00 |        70 
  (1, 6)  |         5M |          0.9 |            13.6 |        1.42h (1/CPU-2) |          1.00 |        55 
  (1, 7)  |        20M |          0.8 |             3.7 |       17.54h (1/CPU-3) |          0.97 |        39 
  (1, 7)  |        20M |          0.9 |             3.7 |       16.17h (1/CPU-3) |          1.00 |        33 
  (1, 7)  |        20M |          1.0 |             3.7 |       20.07h (1/CPU-3) |          1.00 |        31
  (1, 9) (*2)|    750M |          1.0 |              91 |    approx 8w (8/CPU-2) |          0.26 |        23 
  (5, 5)  |       600k |          0.8 |             5.7 |     16.95min (1/CPU-2) |          0.85 |       195 
  (5, 5)  |       600k |          0.9 |             4.9 |     16.13min (1/CPU-2) |          0.88 |       191 
  (5, 5)  |         1M |          1.0 |             4.6 |     14.59min (1/CPU-2) |          0.98 |        93 
  (6, 6)  |         5M |          0.8 |            27.8 |        2.86h (1/CPU-2) |          0.78 |        58 
  (6, 6)  |         5M |          0.9 |            27.7 |        2.62h (1/CPU-2) |          0.83 |        58 
  (6, 6)  |         5M |          1.0 |            27.0 |        2.50h (1/CPU-2) |          0.75 |        53 
  (7, 7)  |        40M |          0.8 |            18.7 |       1.11d (10/CPU-3) |          0.62 |       100 
  (7, 7)  |        40M |          0.9 |            18.7 |      23.38h (10/CPU-3) |          0.68 |       100 
  (7, 7)  |        40M |          1.0 |            18.7 |      17.21h (10/CPU-3) |          0.74 |       100 
  (8, 8)  |       150M |          0.8 |            33.7 |       2.07w (10/CPU-3) |          0.25 |        48 
  (8, 8)  |       150M |          0.9 |            33.7 |       1.59w (10/CPU-3) |          0.33 |        55 
  (8, 8)  |       150M |          1.0 |            33.7 |       1.54w (10/CPU-3) |          0.35 |        49 
  (8, 8)  |       300M |          0.8 |            65.5 |        2.73w (8/CPU-2) |          0.30 |        10 
  (8, 8)  |       300M |          0.9 |            65.5 |        1.64w (8/CPU-2) |          0.42 |        26 
  (8, 8)  |       300M |          1.0 |            65.5 |        2.53w (8/CPU-2) |          0.28 |        99 

(*1) Note that memory usage was higher in previous versions, hence the discrepancy from the paper.
(*2) Note that this result was derived and extrapolated from incomplete runs of the learning algorithm. For details see
README-19.md

CPU-2: Intel Xeon® Gold® 6130 at 2.1GHz; CPU-3: Intel® Gold® E5-2630 v4 at 2.2GHz

### Attack Run Time by Challenge Length 

Attack run time for different challenge lengths; times shown refer to time until first success in single-threaded runs. 
Every data point shows the best obtained time until first success for various choices of guessed amounts of 
challenge-response pairs. Interpolations given were computed using regression for (a, b) on (a * n^b) using all 
available data.

![](../../../figures/ipuf.split.n.png?raw=true)

### Attack Run Time by Interpose PUF Size

Attack run time and best-performing number of CRPs for Interpose PUFs of different reliability and varying number of 
employed arbiter chains. Interpolations given were computed using regression for (a, b) on (a * b^k) using all available
data.

![](../../../figures/ipuf.split.size.png?raw=true)

### Accuracy of Learning Results using Logisitic Regression Attack with Unknown Interpose Bit

Initial accuracy of the lower layer model after training using the CRP set with randomly guessed interpose bits for 
(1, k) and (k,k)-Interpose PUFs. Results shown are using the estimated best number of challenge-response pairs (see also
Table above); accuracy is relative to the PUF simulation's reliability. As with ordinary learning of XOR Arbiter PUFs, 
the probability to obtain a high-accuracy model of the lower layer depends on the size of the Interpose PUF and the 
training set size.

(Just for completeness, our results are artificially capped at 95% due to termination-criteria of the algorithm which 
increase performance. For models where the initial modeling resulted in the model for a half-inverted lower layer, the 
accuracy on this is shown.)

![](../../../figures/ipuf.split.initial.png?raw=true)
    


