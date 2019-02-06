Breaking the Lightweight Secure PUF: Understanding the Relation of Input Transformations and Machine Learning Resistance
==============

pypuf contains three studies that were used in Breaking the Lightweight Secure PUF.


Figure 3: Success Rate of LR Attack on the Pseudorandom Input Transformations
--------------

Success rate of logistic regression attacks on simulated XOR Arbiter
PUFs with 64-bit arbiter chains and four arbiter chains each, based on at least
250 samples per data point shown. Accuracies better than 70% are considered
success (but cf. Figure 4). Four different designs are shown: of the four arbiter
chains in each instance, an input transform is used that transforms zero, one,
two, and three challenges pseudorandomly, keeping the remaining challenges
unmodified. The success rate decreases when the number of arbiter chains with
pseudorandom challenges is increased. The case with 4 pseudorandom sub-challenges
is not shown as it coincides with the results for 3 pseudorandom
challenges. Note the log-scale on the x-axis.

![](../../../figures/breaking_lightweight_secure.success_rates.fig_03.png?raw=true "Breaking Lightweight Secure Fig 3")

Data and figure can be generated with

    python3 -m study breaking_lightweight_secure.success_rates


Figure 4: Accuracy Distribution of LR and Correlation Attack
--------------

Accuracy distribution for learning attempts on randomly chosen
simulated XOR Arbiter PUF instances with different input transformations.
All experiments were run on 64-bit, 4-XOR Arbiter PUFs. When using the
Lightweight Secure input transformation, some learning attempts end with an
intermediate result, while both classic XOR Arbiter PUF and pseudorandom
sub-challenges do not show intermediate solutions. It can be seen that
using our new correlation attack, the resulting model accuracy is increased
significantly over the plain LR attack.

![](../../../figures/breaking_lightweight_secure.accuracy_distribution.fig_04.png?raw=true "Breaking Lightweight Secure Fig 4")

Data and figure can be generated with

    python3 -m study breaking_lightweight_secure.accuracy_distribution


Table II: Attack Run Time
--------------

Expected time until the first success for LR and correlation attacks on classic XOR Arbiter PUF,
Lightweight Secure XOR Arbiter PUF, and Permutation-Based XOR Arbiter PUF. A prediction accuracy
of at least 98% is considered success. Run times refer to single-threaded runs on Intel® Xeon® Gold 6130 CPUs.
All entries are based on 1000 and 600 samples for n=64 and n=128, respectively.

 n  |  k  |     #CRPs  | LR on Classic | LR on Lw Secure     | Corr. Attack on Lw Secure  | LR on Permutation-Based
--- | --- | ---------- | ------------- | ------------------- | -------------------------- | -----------------------
 64 |  4  |     12,000 |        0m 33s |             10m 11s |                     0m 58s |                 24m 50s
 64 |  4  |     30,000 |        0m 31s |              3m 57s |                     0m 44s |                  4m 45s
 64 |  5  |    300,000 |        7m 03s |              3h 03m |                    11m 07s |                 13h 59m
 64 |  6  |  1,000,000 |       42m 30s |              8 days |                     1h 42m |     longer than 96h 00m
 64 |  7  |  2,000,000 |       75h 07m | longer than 20 days |                     8 days |     longer than 16 days
128 |  4  |  1,000,000 |       20m 31s |              2h 53m |                    51m 23s |                 58m 38s
128 |  5  |  2,000,000 |        1h 35m |             35h 20m |                     3h 17m |     longer than 16 days

Data, table and figure can be generated with

    python3 -m study breaking_lightweight_secure.attack_runtime
    

Figure 5: Accuracy Distribution of LR and Correlation Attack
--------------

When ordering possible permutations for the correlation attack by
validation set accuracy, the target permutation often appears within the first
few permutations. To obtain 80% probability of finding the permutation among
the first tested, we chose to check the first four permutations for k = 4, the
first ten permutations for k = 5 and the first 18 permutations for k = 6. This
justifies restarting the logistic regression learner for only a couple possible
permutations instead of all k! many.

![](../../../figures/breaking_lightweight_secure.attack_runtime.fig_05.png?raw=true "Breaking Lightweight Secure Fig 5")

Data and figure can be generated with

    python3 -m study breaking_lightweight_secure.attack_runtime


Figure 6: Success Rate of LR Attack on Classic, Lightweight Secure, Permutation, and Pseudorandom Input Transformation
--------------

Success rate of logistic regression attacks on simulated XOR
Arbiter PUFs with 64-bit arbiter chains and four arbiter chains each. Four
different input transformations are shown: classic, Lightweight Secure by
Majzoobi et al., a permutation-based input transformation, and a pseudorandom
input transformation used as comparison.
All data points are based off at least 100 samples. For a success threshold
of 70%, Lightweight Secure and classic are equally hard to attack, whereas
permutation-based and pseudorandom require significantly more CRPs.

![](../../../figures/breaking_lightweight_secure.success_rates.fig_06.png?raw=true "Breaking Lightweight Secure Fig 6")

Data and figure can be generated with

    python3 -m study breaking_lightweight_secure.success_rates
