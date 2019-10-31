# Cryptanalysis of the Interpose PUF

This study analyzes the Interpose PUF proposed by Nguyen et al., ePrint of 2018, presented at CHES 2019.

## Approximating Arbiter PUFs by their Chow Parameters

It is clear that any LTF `f(x) = sgn p(x)` is uniquely identified by its weights `w`,
where `p(x) = 〈w,x〉`. Perhaps surprisingly, an LTF is also uniquely identified by its
Chow parameters, i.e. the Fourier coefficients of the first degree (plus the bias value,
if any).

Generalizing, any PTF `f(x) = sgn p(x)` where `p` is some polynomial over `{-1,1}`,
in uniquely identified by the Fourier coefficients of `f` on the parities that appear
in the polynomial `p`.

In general, the Chow parameters are bad estimations of the weights.
But how do they estimate the weights if the weights are chosen from a Gaussian
distribution?


### Chow-Parameters are Good Estimators for Arbiter PUFs

We study two approximations for the XOR Arbiter PUF.

1. Take the original XOR Arbiter PUF `f(x) = sgn Π p(w,x)` and replace the weights `w` with the Chow parameters,
    resulting in approximation `f'(x) = sgn Π p(chow,x)` where `chow` are the Chow parameters. We define the 
    quality of this approximation *accuracy*, defined as the probability over all inputs `x` that `f(x) == f'(x)`. 
2. Same as 1, but do not apply the `sgn` in the approximation. The approximated value will therefore be a real value.
    We define the quality of this approximation *distance*, defined as the expected value of `|f(x) - f'(x)| / 2`
    over all inputs `x`. (Note that this notion is a generalization of the accuracy defined in 1.)

Empirically, we can show that both approximations have high quality for the usual values of `n` and `k`, with the 
quality decreasing with increasing `n` and `k`.

For approximation 1, we obtain the following accuracies (higher means better quality)

![](../../../figures/ipuf.approximation.xor-arbiter-puf.accuracy.png?raw=true "Chow Parameter Approximation for XOR Arbiter PUFs")

For approximation 2, we obtain the following distances (*lower* means better quality)

![](../../../figures/ipuf.approximation.xor-arbiter-puf.dist.png?raw=true "Chow Parameter Approximation for XOR Arbiter PUFs")

Additionally we note that the quality of approximation 2 seems to be independent of parameter `n`. How can these 
approximations be used to obtain an Interpose PUF approximation?


## Approximating the Interpose PUF

As the Interpose PUF consists of two XOR Arbiter PUFs (called "up" and "down"), we need to choose for each one how to
approximate it. Here, we choose to use approximation 2 for the "up" XOR Arbiter PUF and model the "down" XOR Arbiter PUF
directly using it's weights. (Why we choose this particular mode of approximation will become clear during the attack.)

As can be expected, the further processing of the approximated value of the "up" XOR Arbiter PUF will also further
decrease the overall quality of approximation. However, for some values of `k_up`, `k_down`, the approximation still
yields useful results.

![](../../../figures/ipuf.approximation.heatmap.png?raw=true "Chow Parameter Approximation for XOR Arbiter PUFs")

To study the quality of the approximation, a white box approach was chosen. The Chow parameters of the "up" XOR Arbiter
PUF are computed chain by chain, effectively modeling each chain. (Using this method, the number of Fourier 
coefficients to approximate is `n·k_up` as opposed to `O(n^k_up)` when estimating Chow parameters of the combined `k_up`
chains.)

The estimated value of the "up" XOR Arbiter PUF is then, without taking the sign, fed into the "down" XOR Arbiter PUF,
which is modelled by the original simulation model, using the original weights.

All data and figures can be generated with

    python3 -m study ipuf.approximation

Runtime is on the order of minutes.


## Attacking the Interpose PUF

Using the approximation outlined and studied above, an `n`-bit (`k_up`,`k_down`)-Interpose PUF can be approximated by
a (much) larger `m`-bit `k_down`-XOR Arbiter PUF.

We observed that, as we use approximation 2 for the "up" XOR Arbiter PUF of the target Interpose PUF, no sign function
is involved. Therefore, the approximation `f_up'(x) = sgn Π p(chow,x)` can be plugged into the traditional delay value
model of the "down" XOR Arbiter PUF and be expanded, resulting in an "XOR Arbiter PUF" with `m`-bit challenges, where
the value `m` depends on `n` and `k_up` and may be very large.  

![](../../../figures/ipuf.ptf_length.png?raw=true "Chow Parameter Approximation for XOR Arbiter PUFs")

Preliminary results show that the approach is train-able, with accuracies up to the limitation by the approximation.
However, the large models require a massive training set size and hence a long training time.

![](../../../figures/ipuf.lr_attack.png?raw=true "Model based on Chow Parameter Approximation for XOR Arbiter PUFs, Trained with Logistic Regression")
