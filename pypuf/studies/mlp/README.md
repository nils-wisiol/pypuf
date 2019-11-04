Learning XOR Arbiter Physically Unclonable Functions using Multilayer Perceptrons
==============

This study examines the learning behavior of Multilayer Perceptrons on
XOR Arbiter PUFs. It nearly replicates the learning results on 64-Bit
k-XOR Arbiter PUFs from Aseeri et al., "A Machine Learning-based
Security Vulnerability Study on XOR PUFs for Resource-Constraint
Internet of Things", IEEE International Congress on Internet of Things
(ICIOT), San Francisco, CA, pp. 49-56, 2018. It also shows that the
runtime of our experiments where the width k of PUF simulations equals 7
or 8 highly differs from those of Aseeri et al.


For each k in {4, 5, 6, 7, 8} 200 different 64-Bit k-XOR APUFs were
learned by an MLP, where each 100 experiments used the Tensorflow
implementation and 100 used the Scikit-learn implementation,
respectively. Every experiment is abstracted to its accuracy and plotted
as one point within the corresponding scatter plot. All
Scikit-learn results are visualized on the left side while the
Tensorflow results are on the right side. Note the logarithmic scale.
The corresponding reference values from Aseeri et al. are plotted as black
lines.

In the second row, the run time of these experiments are shown.
The scale on the
y-axes is logarithmic using powers of 10. The reference values from Aseeri et al.
are shown in black.

![](../../../figures/mlp.aseeri.png "Runtime Overview of MLP on (64, k)-XOR APUFs")

Data and figures can be generated with

    python3 -m study mlp.aseeri
