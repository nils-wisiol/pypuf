# pypuf: Python PUF Simulator and Learner ![Master Branch Build Status Indicator](https://travis-ci.org/nils-wisiol/pypuf.svg?branch=master)

Physically Unclonable Functions (PUFs) are of research interest in the field of lightweight and secure authentication.
PUFs usually provide a challenge-response-interface.
It is of fundamental importance to study the security of this interface.
One way to study the security is per empirical study.
(Remember though, that this can only prove insecurity of any given PUF.)

pypuf provides simulations and attacks on PUFs; it aims at helping to understand PUFs and attacks on them.
It also provides some tools for running experiments with the simulations and attacks.

Technically, pypuf heavily relies on numpy.
Some operations base on an additional module that has been written in C (see installation section below).

## Installation

Currently pypuf relies heavily on `numpy` and is tested for Python versions 3.6 and 3.7.
Some operations also require the [polymath](https://github.com/taudor/polymath) package and/or the [scipy](https://www.scipy.org/) package.

### Recommended Installation

The recommended way to run pypuf is to use a virtual environment.
It can be created using the following steps.
Please make sure that virtualenv, python, the python development files,
and a compiler toolchain are available on your system.
(On Debian: `apt install build-essential python3 python3-dev virtualenv`.)

In your pypuf clone directory,

    # create and enter virtual environment
    virtualenv -p python3 env
    source env/bin/activate

    # upgrade pip
    python3 -m pip install --upgrade pip

    # install requirements (polymath needs c99)
    CC="gcc -std=c99" pip3 install -r requirements.txt

Afterwards, confirm a correct setup by running the tests:

    python3 -m unittest

If you encounter any trouble, please refer to our continuous integration at [travis-ci](https://travis-ci.org/nils-wisiol/pypuf)
to see a working example or raise an issue on GitHub.

### Lazy Installations

You can run pypuf installing numpy and scipy from your distribution's repository.
This will prevent you from using any features that rely on `polymath` and is hence not recommended.
It is an easier way however to get started quickly.
After installing `python3`, `numpy`, and `scipy` run the example to make sure everything is setup okay.
(Unit tests for features relying on `polymath` will fail in this scenario!)

## Idea

pypuf mainly consists of two parts, simulation and learning.
Also, there is the 'administrative' toolset and experiment scheduler built around those two core parts.

Note that pypuf uses the {-1,1} notation of bits, where True = -1 and False = +1
(that is, -1 corresponds to traditional "1" and +1 corresponds to traditional "0").

### Simulation

The simulation currently consists of a very broad class, the LTF Array Simulator.
It can simulate an array of Linear Threshold Functions and hence simulate [Arbiter PUFs](https://people.csail.mit.edu/devadas/pubs/cpuf-journal.pdf),
XOR Arbiter PUFs, [Lightweight Secure PUFs](http://aceslab.org/sites/default/files/Lightweight%20Secure%20PUFs_0.pdf),
Majority Vote PUFs, and more custom designs.
To that end, the input transformation can be chosen (e.g. as designed for the Lightweight Secure PUF)
and the combiner function can be chosen (to generalize the usually used XOR function).  
Another component of the simulation is the Fourier expansion of a Boolean function.
It either can be evaluated returning a real value or boxed into the sign operator, returning -1 or +1.

#### Input Transformation

`LTFArray` currently defines a couple of input transformations.
More input transformations can be added by implementing a function and provide the function as a constructor argument to `LTFArray`.

 * `id`: use the generated challenge directly as input to the LTF (note that this *does not* correspond to the physical implementation of Arbiter PUFs)
 * `atf`: use *Arbiter Threshold Functions*, that is, transform the challenges in a way such that we simulate physical implementations of Arbiter PUFs
 * `lightweight_secure_original`: input transformation as defined by Majzoobi et al. in [Lightweight Secure PUFs](http://aceslab.org/sites/default/files/Lightweight%20Secure%20PUFs_0.pdf) 
 * `soelter_lightweight_secure`: as defined by Majzoobi et al., but with a one-bit modification due to Sölter.
 * `polynomial`: challenges are interpreted as polynomials from GF(2^n).
    From the initial challenge c the i-th Arbiter chain gets the coefficients of the polynomial c^(i+1) as challenge.
    Only challenges with length 8, 16, 24, 32, 48, 64 are accepted.
 * `permutation_atf`: for each Arbiter chain first a pseudorandom permutation is applied and thereafter the ATF transform.
 * `random`: Each Arbiter chain gets a random challenge derived from the original challenge using a PRNG.

 `LTFArray` also implements "input transformation generators" that can be used to combine existing input transformations into new ones.
 * `generate_concatenated_transform(transform_1, nn, transform_2)`:
    the first `nn` bit will be transformed using `transform_1`, the rest will be transformed with `transform_2`. 
 * `generate_random_permutation_transform(seed, nn, kk, atf)`: each of the `kk` LTFs will be fed a random,
    but fixed permutation generated based on the given seed.
    If `atf`, then challenges will be ATF-transformed after permuting them.
 * `generate_stacked_transform(transform_1, kk, transform_2)`: the first `kk` challenges will be transformed using `transform_1`,
    the rest will be transformed with `transform_2`.

#### Combiner Function

`LTFArray` currently provides the traditional XOR (that is, parity) as a combiner function,
as well as the Inner Product Mod 2 function.
Further combiner functions can be implemented as static functions in `LTFArray`,
or anywhere else and given to the `LTFArray` constructor.

### Learning

pypuf currently ships a logistic regression algorithm that was proposed to learn (XOR) Arbiter PUFs by [Sölter](https://www.researchgate.net/profile/Jan_Soelter/publication/259580784_Cryptanalysis_of_electrical_PUFs_via_machine_learning_algorithms/links/00b4952cc03621836c000000/Cryptanalysis-of-electrical-PUFs-via-machine-learning-algorithms.pdf)
and [Rührmair et al](https://eprint.iacr.org/2010/251.pdf), utilizing the RPROP backpropagation.
Additionally pypuf aims for the provision of PAC learning algorithms,
currently represented only by the Low Degree Algorithm introduced by [Mansour](http://www.cs.columbia.edu/~rocco/Teaching/S12/Readings/Mansour-survey.pdf).

## Usage

pypuf is primarily designed as an API. However, it provides a subset of its features as a command line interface.

### Command Line Interface

`sim_learn` is a command line interface that simulates an LTF array and tries to learn it using logistic regression.
For simulation and learning, a couple of parameters can be chosen on the command line.
Start `sim_learn` without parameters to get detailed usage information.

`mv_num_of_votes` is a command line interface that allows to compute the minimum number of required votes in a Majority Vote XOR Arbiter PUF
such that a certain stability is achieved.
For details, please refer to [Why Attackers Lose](https://eprint.iacr.org/2017/932.pdf).

`stability_calculation` is a command line interface that allows to generate a stability histogram for a simulated PUF,
i.e. to determine an approximation of how the probability to see the 'correct' answer is distributed among the challenge space.
For details, please refer to [Why Attackers Lose](https://eprint.iacr.org/2017/932.pdf).

#### Example Usage

Example usage of `sim_learn` that simulates a 64 bit 2-xor Arbiter PUF and learns it to approx. 98% from 12000 challenge response pairs:
`python3 sim_learn.py 64 2 atf xor 12000 1 1 0xdead 0xbeef`

### API

pypuf's two most important interfaces are the `Learner` and the `Simulation` that can interact with each other.

#### Example Usage

This example creates a 64 bit 2-XOR XOR Arbiter PUF `instance` and a Logistic Regression learner `learner_lr` to learn it.
After learning, the model's accuracy is tested and the result is printed.

To run, use `python3 example.py`.

````python
from pypuf import tools
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray

# create a simulation with random (Gaussian) weights
# for 64-bit 2-XOR
instance = LTFArray(
    weight_array=LTFArray.normal_weights(n=64, k=2),
    transform=LTFArray.transform_atf,
    combiner=LTFArray.combiner_xor,
)

# create the learner
lr_learner = LogisticRegression(
    t_set=tools.TrainingSet(instance=instance, N=12000),
    n=64,
    k=2,
    transformation=LTFArray.transform_atf,
    combiner=LTFArray.combiner_xor,
)

# learn and test the model
model = lr_learner.learn()
accuracy = 1 - tools.approx_dist(instance, model, 10000)

# output the result
print('Learned a 64bit 2-xor XOR Arbiter PUF from 12000 CRPs with accuracy %f' % accuracy)
````

## Contribution

All contributions receive warm welcome!
To contribute, simply open an issue and/or pull request and maintain coding style.
Please follow the coding standards within this project, that is, keep your code just like the code that is already there.
Also, please don't forget to add tests for your code.
We will only accept contributions under GNU GPLv3.

If you're using pypuf in your research, please let us know so we can link to your work here.

### Contribution quick check list

 * Is your contribution GPLv3 compatible?
 * Update README.md accordingly
 * Document new code, update code comments for changed code
 * Provide tests for your code (to run the tests locally, use `python3 -m unittest`)
 * Do not use `numpy.random` directly; always use an `numpy.random.RandomState` instance.

## Authors

Significant contribution to this project are due to (in chronological order):

 * Nils Wisiol (nils.wisiol++a+t++fu-berlin.de)
 * Christoph Graebnitz (christoph.graebnitz++a+t++fu-berlin.de)
 * Christopher Mühl (chrism++a+t++zedat.fu-berlin.de)
 * Benjamin Zengin (benjamin.zengin++a+t++fu-berlin.de)
