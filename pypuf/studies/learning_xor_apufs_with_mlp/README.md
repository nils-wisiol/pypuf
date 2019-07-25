Learning XOR Arbiter PUFs using a Multilayer Perceptron
==============

pypuf contains two studies that were used to analyze the learning behavior of
Multilayer Perceptrons on XOR Arbiter PUFs.


Figure 1: Learning Results from Aseeri et. al. on 64-Bit k-XOR Arbiter PUFs in Comparison to our Replicates regarding Accuracy
--------------

Description of Figure 1...

![](../../../figures/mlp_aseeri_overview_accuracy.pdf?raw=true "Overview MLP Aseeri et. al. Accuracy")


Figure 2: Learning Results from Aseeri et. al. on 64-Bit k-XOR Arbiter PUFs in Comparison to our Replicates regarding Runtime
--------------

Description of Figure 2...

![](../../../figures/mlp_aseeri_overview_runtime.pdf?raw=true "Overview MLP Aseeri et. al. Runtime")

Data and figures can be generated with

    python3 -m study mlp_aseeri_et_al.py



Figure 3: Learning Results on 1-, 2-, and 3-XOR Arbiter PUFs with various Lengths regarding Accuracy
--------------

Description of Figure 3...

![](../../../figures/mlp_various_lengths_accuracy.pdf?raw=true "Overview MLP Aseeri et. al.")


Figure 4: Learning Results on 1-, 2-, and 3-XOR Arbiter PUFs with various Lengths regarding Runtime
--------------

Description of Figure 4...

![](../../../figures/mlp_various_lengths_runtime.pdf?raw=true "Overview MLP Aseeri et. al.")

Data and figure can be generated with

    python3 -m study mlp_aseeri_et_al.py
