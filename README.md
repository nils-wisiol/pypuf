# pypuf: Python PUF Simulator and Learner ![Master Branch Build Status Indicator](https://travis-ci.org/nils-wisiol/pypuf.svg?branch=master)

Physically Unclonable Functions (PUFs) are of research interest in the field of lightweight and secure authentication. PUFs usually provide a challenge-response-interface. It is of fundamental importance to study the security of this interface. One way to study the security is per empirical study. (Remember though, that this can only prove the insecurity of a given PUF.)

## Installation

pypuf solely needs python3 and `numpy`. Hence, we do not rely on a virtual environment as of now.

## Idea

pypuf mainly consists of two parts, simulation and learning. Also, there is the 'administrative' toolset and experiment scheduler built around those two core parts.

Note that pypuf uses the {-1,1} notation of bits, where True = -1 and False = +1 (that is, -1 corresponds to traditional "1" and +1 corresponds to traditional "0").

### Simulation

The simulation currently consists of just one very broad class, the LTF Array Simulator. It can simulate an array of Linear Threshold Functions and hence simulate [Arbiter PUFs](https://people.csail.mit.edu/devadas/pubs/cpuf-journal.pdf), XOR Arbiter PUFs, [Lightweight Secure PUFs](http://aceslab.org/sites/default/files/Lightweight%20Secure%20PUFs_0.pdf), and more custom designs. To that end, the input transformation can be chosen (e.g. as designed for the Lightweight Secure PUF) and the combiner function can be chosen (to generalize the usually used XOR function).

#### Input Transformation

`LTFArray` currently defines a couple of input transformations. More input transformations can be added by implementing a function and provide the function as a constructor argument to `LTFArray`.

 * `id`: use the generated challenge directly as input to the LTF (note that this *does not* correspond to the physical implementation of Arbiter PUFs)
 * `atf`: use *Arbiter Threshold Functions*, that is, transform the challenges in a way such that we simulate physical implementations of Arbiter PUFs
 * `mm`: experimental input transformation with long PTF representation
 * `lightweight_secure`: input transformation as defined by Majzoobi et al. in [Lightweight Secure PUFs](http://aceslab.org/sites/default/files/Lightweight%20Secure%20PUFs_0.pdf)

#### Combiner Function

`LTFArray` currently only provides the traditional XOR (that is, parity) as a combiner function. Further combiner functions can be implemented as static functions in `LTFArray`, or anywhere else and given to the `LTFArray` constructor.

### Learning

pypuf currently ships a logistic regression algorithm that was proposed to learn (XOR) Arbiter PUFs by [Sölter](https://www.researchgate.net/profile/Jan_Soelter/publication/259580784_Cryptanalysis_of_electrical_PUFs_via_machine_learning_algorithms/links/00b4952cc03621836c000000/Cryptanalysis-of-electrical-PUFs-via-machine-learning-algorithms.pdf) and [Rührmair et al](https://eprint.iacr.org/2010/251.pdf).

## Usage

pypuf is primarily designed as an API. However, it provides a subset of its features as a command line interface.

### Command Line Interface

`sim_learn` is a command line interface that simulates an LTF array and tries to learn it using logistic regression. For simulation and learning, a couple of parameters can be chosen on the command line. Start `sim_learn` without parameters to get detailed usage information.	

#### Example Usage

Example usage of `sim_learn` that simulates a 64 bit 2-xor Arbiter PUF and learns it to approx. 98% from 12000 challenge response pairs: `python3 sim_learn.py 64 2 atf xor 12000 1 1 0xdead 0xbeef`

### API

pypuf ships to important interfaces `Learner` and `Simulation` that can then interact with each other.

#### Example Usage

This example creates a 64 bit 2-XOR XOR Arbiter PUF `instance` and a Logistic Regression learner `learner_lr` to learn it. After learning, the model's accuracy is tested and the result is printed.

````python
from pypuf import simulation, learner, tools

# create a simulation with random (Gaussian) weights
# for 64-bit 4-XOR 
instance = simulation.LTFArray(
    weight_array=simulation.LTFArray.normal_weights(n=64, k=2),
    transform=simulation.LTFArray.transform_atf,
    combiner=simulation.LTFArray.combiner_xor,
)

# create the learner
lr_learner = learner.LogisticRegression(
    training_set=tools.TrainingSet(instance=instance, N=12000),
    n=64,
    k=2,
    transformation=simulation.LTFArray.transform_atf,
    combiner=simulation.LTFArray.combiner_xor,
)

# learn and test the model
model = lr_learner.learn()
accuracy = 1 - tools.approx_dist(instance, model, 10000)

# output the result
print('Learned a 64bit 2-xor XOR Arbiter PUF from 12000 CRPs with accuracy %f' % accuracy)
````

## Contribution

All contributions receive warm welcome! Please obey the coding standards within this project, that is, keep your code just like the code that is already there. Also, please don't forget to add tests for your code. We can only accept contributions under GNU GPLv3.

If you're using pypuf in your research, please let us know so we can link to your work here.

### Contribution quick check list

 * Is your contribution GPLv3 compartible?
 * Update README.md accordingly
 * Document new code, update code comments for changed code
 * Provide tests for your code
 * Do not use `numpy.random` directly; always use an `numpy.random.RandomState` instance.

## Authors

Significant contribution to this project are due to (in chronological order):

 * Nils Wisiol (nils.wisiol++a+t++fu-berlin.de)
 * Christoph Graebnitz (christoph.graebnitz++a+t++fu-berlin.de)
 * Christopher Mühl (chrism++a+t++zedat.fu-berlin.de)