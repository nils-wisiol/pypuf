# Property Testing

"Property Testing is concerned with the computational task of determining whether
a given object has a predetermined property or is “far” from any object having
the property." Oded Goldreich, Shari Goldwasser, and Dana Ron. 1998. Property testing
and its connection to learning and approximation. J. ACM 45, 4 (July 1998), 653-750. 
DOI=http://dx.doi.org/10.1145/285055.285060

This Module implements several functions to determine properties of `pypuf.simulation.base.Simulation`.
The easiest and usually calculated characteristics of PUF instances in literature are the reliability
and uniqueness. These functions are more statistical functions rather than function properties.
Nevertheless, these metrics are essential for every PUF Class, and a PUF instance should meet
their ideal values. These strategies are naive but can reveal unacceptable attributes.
## Reliability
In literature, the reliability of a PUF instance is described as the arithmetic
mean of the hamming distances of the repeated evaluations of the same challenge
[1], [2], [20]. There are publications which investigate environmental variations
such as temperature [4], voltage [5] and aging [6]. These physical occurrences
are known to have an impact on certain PUF instances. An author might be
interested in characterizing the reliability and uniqueness of a PUF instance
under specific occurrences. In practical situations, it is common to approximate 
the actual response P(c) of a PUF instance majority voting over a set of evaluation results. 
An optimal value of reliability for a noiseless PUF is 0% which means that all responses to 
the same challenge were the same.

The reliability for a challenge <img src="https://latex.codecogs.com/svg.latex?c\in\left\{&space;-1,1\right\}&space;^{n}" title="c\in\left\{ -1,1\right\} ^{n}" /> and <img src="https://latex.codecogs.com/svg.latex?r" title="r" /> evaluations of a PUF instance 
<img src="https://latex.codecogs.com/svg.latex?\mathrm{P}:\{-1,1\}^{n}\rightarrow\{-1,1\}^{\eta}" title="\mathrm{P}:\{-1,1\}^{n}\rightarrow\{-1,1\}^{\eta}" /> is
<img src="https://latex.codecogs.com/svg.latex?\mathsf{\mathrm{R}}(\mathrm{P},c,r)=\frac{1}{r}\sum_{j=1}^{r}\frac{\mathrm{H}(\mathrm{P}(c),\mathrm{P}^{j}(c))}{\eta}\times100\%" />

where <img src="https://latex.codecogs.com/svg.latex?\mathrm{P^{(j)}(c)}" title="\mathrm{P^{(j)}(c)}" /> is the j-th evaluation result and <img src="https://latex.codecogs.com/svg.latex?\mathrm{P(c)}" title="\mathrm{P(c)}" /> is the actual response.

As we only support instances which generate a single bit response we just use the Hamming distance instead of the fractional hamming distance.
We also did not store the reliability as percentage values.
Additionally, to the standard reliability calculation, we provide an extended statistic (`PropertyTest.reliability_statistic`) which offers more information about the minimum, maximum and median of the reliabilities for a set of simulation instances, a set of challenges and a number of evaluations.

### Examples

Reliability calculation:
```python
def example_reliability():
    """This method shows how to use the PropertyTest.reliability function."""
    n = 8
    k = 8
    transformation = NoisyLTFArray.transform_id
    combiner = NoisyLTFArray.combiner_xor
    weights = NoisyLTFArray.normal_weights(n=n, k=k)
    instance = NoisyLTFArray(
        weight_array=weights,
        transform=transformation,
        combiner=combiner,
        sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(n, 0.5)
    )
    challenge = array([-1, 1, 1, 1, -1, 1, 1, 1])
    reliability = PropertyTest.reliability(instance, reshape(challenge, (1, n)))
    print('The reliability is {}.'.format(reliability))
```
Reliability statistic calculation:
```python
def example_reliability_statistic():
    """This method shows hot to use the PropertyTest.reliability_statistic."""
    n = 8
    k = 1
    N = 2 ** n
    instance_count = 3
    measurements = 100
    transformation = NoisyLTFArray.transform_id
    combiner = NoisyLTFArray.combiner_xor
    weights = NoisyLTFArray.normal_weights(n=n, k=k)
    instances = [
        NoisyLTFArray(
            weight_array=weights,
            transform=transformation,
            combiner=combiner,
            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(n, 0.5)
        ) for _ in range(instance_count)
    ]
    challenges = array(list(sample_inputs(n, N)))
    property_test = PropertyTest(instances)
    reliability_statistic = property_test.reliability_statistic(challenges, measurements=measurements)
    print('The reliability statistic is {}.'.format(reliability_statistic))
```
## Uniqueness
Likewise, it must be mentioned how to calculate the uniqueness of a PUF. For
this purpose, it is necessary to start with the common definition of uniqueness.
The uniqueness of a PUF is described by the average response inter-distance
between different PUF instances.
For a challenge <img src="https://latex.codecogs.com/svg.latex?c\in\left\{&space;-1,1\right\}&space;^{n}" title="c\in\left\{ -1,1\right\} ^{n}" /> and <img src="https://latex.codecogs.com/svg.latex?\mathbb{P}" title="\mathbb{P}" /> a set of PUF instances <img src="https://latex.codecogs.com/svg.latex?\mathrm{P}:\{-1,1\}^{n}\rightarrow\{-1,1\}^{\eta}" title="\mathrm{P}:\{-1,1\}^{n}\rightarrow\{-1,1\}^{\eta}" /> the uniqueness is expressed as

<img src="https://latex.codecogs.com/svg.latex?\mathrm{U}(\mathbb{P},c)=\frac{2}{m(m-1)}\sum_{u=1}^{m-1}\sum_{v=u&plus;1}^{m}\frac{\mathrm{H}(R_{u}(c),R_{v}(c))}{\eta}\times100\%" title="\mathrm{U}(\mathbb{P},c)=\frac{2}{m(m-1)}\sum_{u=1}^{m-1}\sum_{v=u+1}^{m}\frac{\mathrm{H}(R_{u}(c),R_{v}(c))}{\eta}\times100\%" />

where <img src="https://latex.codecogs.com/svg.latex?R=(\mathbb{P}_{1}(c),\ldots,\mathbb{P}_{i}(c),\ldots,\mathbb{P}_{m}(c))" title="R=(\mathbb{P}_{1}(c),\ldots,\mathbb{P}_{i}(c),\ldots,\mathbb{P}_{m}(c))" /> is a tuple of the responses of PUF
instances which can vary between repeated calculations of <img src="https://latex.codecogs.com/svg.latex?\mathrm{U}(\mathbb{P},c)" title="\mathrm{U}(\mathbb{P},c)" />.
An optimal uniqueness value for a set of PUF instances is 50%. Like the reliability the uniqueness is also not stored as percentage.
The statistics applied to the reliability can also be used to the uniqueness.

### Examples
Uniqueness calculation:
```python
def example_uniqueness():
    """
    This method shows the function which can be used to calculate the uniqueness of a set of simulation instances.
    """
    n = 8
    k = 1
    instance_count = 3
    transformation = NoisyLTFArray.transform_id
    combiner = NoisyLTFArray.combiner_xor
    weights = NoisyLTFArray.normal_weights(n=n, k=k)
    instances = [
        NoisyLTFArray(
            weight_array=weights,
            transform=transformation,
            combiner=combiner,
            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(n, weights)
        ) for _ in range(instance_count)
    ]
    challenge = array([-1, 1, 1, 1, -1, 1, 1, 1])
    uniqueness = PropertyTest.uniqueness(instances, reshape(challenge, (1, n)))
    print('The uniqueness is {}.'.format(uniqueness))
```
Uniqueness statistic calculation:
```python
def example_uniqueness_statistic():
    """This method shows the uniqueness statistic function."""
    n = 8
    k = 1
    N = 2 ** n
    instance_count = 11
    measurements = 1
    transformation = NoisyLTFArray.transform_id
    combiner = NoisyLTFArray.combiner_xor
    weights = NoisyLTFArray.normal_weights(n=n, k=k)
    instances = [
        NoisyLTFArray(
            weight_array=weights,
            transform=transformation,
            combiner=combiner,
            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(n, weights)
        ) for _ in range(instance_count)
    ]

    challenges = array(list(sample_inputs(n, N)))
    property_test = PropertyTest(instances)
    uniqueness_statistic = property_test.uniqueness_statistic(challenges, measurements=measurements)
    print('The uniqueness statistic is {}.'.format(uniqueness_statistic))
```
#### Sources
[1] Q Chen, G Csaba, P Lugli, U Schlichtmann, and U Rührmair. The bistable
ring PUF: A new architecture for strong physical unclonable functions. In
2011 IEEE International Symposium on Hardware-Oriented Security and
Trust, pages 134–141, June 2011.

[2] Tauhidur Rahman, Domenic Forte, Jim Fahrny, and Mohammad Tehra-
nipoor. ARO-PUF: An aging-resistant ring oscillator PUF design. In Pro-
ceedings of the Conference on Design, Automation & Test in Europe, DATE
’14, pages 69:1–69:6, 3001 Leuven, Belgium, Belgium, 2014. European De-
sign and Automation Association.

[3] A Maiti and P Schaumont. Improving the quality of a physical unclonable
function using configurable ring oscillators. In 2009 International Confer-
ence on Field Programmable Logic and Applications, pages 703–707, August
2009.

[4] M Majzoobi, F Koushanfar, and S Devadas. FPGA PUF using pro-
grammable delay lines. In 2010 IEEE International Workshop on Infor-
mation Forensics and Security, pages 1–6, December 2010.

[5] Q Chen, G Csaba, P Lugli, U Schlichtmann, and U Rührmair. Charac-
terization of the bistable ring PUF. In 2012 Design, Automation Test in
Europe Conference Exhibition (DATE), pages 1459–1462, March 2012.

[6] R Maes and V van der Leest. Countering the effects of silicon aging
on SRAM PUFs. In 2014 IEEE International Symposium on Hardware-
Oriented Security and Trust (HOST), pages 148–153, May 2014.
