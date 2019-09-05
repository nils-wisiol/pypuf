Learning XOR Arbiter Physically Unclonable Functions using Multilayer Perceptrons
==============

pypuf contains two studies that were used to analyze the learning
behavior of Multilayer Perceptrons on XOR Arbiter PUFs. The first one
nearly replicates the learning results on 64-Bit k-XOR Arbiter PUFs from
Aseeri et al., "A Machine Learning-based Security Vulnerability Study on
XOR PUFs for Resource-Constraint Internet of Things", IEEE International
Congress on Internet of Things (ICIOT), San Francisco, CA, pp. 49-56,
2018\. It also shows that the runtime of our experiments where the width
k of PUF simulations equals 7 or 8 highly differs from those of Aseeri
et al.

The second study ...


Figure 1: Learning Results of MLPs on 64-Bit k-XOR APUFs in Comparison to those of Aseeri et al. regarding Accuracy
--------------

For each k in {4, 5, 6, 7, 8} 200 different 64-Bit k-XOR APUFs were
learned by an MLP, where each 100 experiments used the Tensorflow
implementation and 100 used the Scikit-learn implementation,
respectively. Every experiment is abstracted to its accuracy and plotted
as one point within the corresponding scatter plot. There are 4
different scatter plots. On the x-axes the data are distinguished
between different widths k, while on the y-axes the data are
distinguished between different accuracies from 50% to 100%. All
Scikit-learn results are visualized on the left side while the
Tensorflow results are on the right side. The plots below show the same
data as their corresponding plot above, but they use a logarithmic scale
on the y-axes in order to enable a better differentiation between high
accuracies. Furthermore, the mean value of the accuracies over all
experiments for each k are plotted as colored lines. Additionally, the
corresponding reference values from Aseeri et al. are plotted as black
lines.


![](../../../figures/learning_xor_apufs_with_mlp.replicate_aseeri_etal_overview_accuracy.png "Accuracy Overview of MLP on (64, k)-XOR APUFs")


Figure 2: Learning Results of MLPs on 64-Bit k-XOR APUFs in Comparison to those of Aseeri et al. regarding Runtime
--------------

The same experiments as visualized in Figure 1 are used for this figure.
But in contrast, every experiment is abstracted to its runtime here
instead of its accuracy and only two scatter plots are shown for each
library that is used for the implementation of MLPs. The scale on the
y-axes is logarithmic using powers of 10. As in Figure 1, the mean
values and reference values are plotted as lines, but additionally the
mean values are connected by a gray line.

![](../../../figures/learning_xor_apufs_with_mlp.replicate_aseeri_etal_overview_runtime.png "Runtime Overview of MLP on (64, k)-XOR APUFs")

Data and figures can be generated with

    python3 -m study learning_xor_apufs_with_mlp.replicate_aseeri_etal



Figure 3: Learning Results on 1-, 2-, and 3-XOR Arbiter PUFs with various Lengths regarding Accuracy
--------------

Description of Figure 3...

![](../../../figures/mlp_various_lengths_accuracy.pdf?raw=true "Overview MLP Aseeri et al.")


Figure 4: Learning Results on 1-, 2-, and 3-XOR Arbiter PUFs with various Lengths regarding Runtime
--------------

Description of Figure 4...

![](../../../figures/mlp_various_lengths_runtime.pdf?raw=true "Overview MLP Aseeri et al.")

Data and figure can be generated with

    python3 -m study learning_xor_apufs_with_mlp.mlp_aseeri_etal
