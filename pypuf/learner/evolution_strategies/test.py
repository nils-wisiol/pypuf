import unittest
import numpy as np
from pypuf import tools
#from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as Becker
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray

class TestLTFArray(unittest.TestCase):

    # create 4 instances of NoisyLTFArray
    prng = np.random.RandomState(seed=0x5000)

    weight_array_1 = np.array([[0.21, -0.7, 2.4, 0.6, 0.1, -0.9], [-2.04, 0.7, 1.2, -0.2, 1.3, -0.8]])
    weight_array_2 = np.array([[1.51, 0.3, -1.2, 0.9, -0.4], [0.17, -2.1, -0.6, 1.4, -0.7]])
    weight_array_3 = LTFArray.normal_weights(32, 4, 1, 3, prng)
    weight_array_4 = LTFArray.normal_weights(64, 8, 0, 1, prng)

    transform = LTFArray.transform_atf
    combiner = LTFArray.combiner_xor

    noise_1 = 0.1 * np.sqrt(np.shape(weight_array_1)[1])
    noise_2 = 0.1 * np.sqrt(np.shape(weight_array_2)[1])
    noise_3 = 0.3 * np.sqrt(np.shape(weight_array_3)[1])
    noise_4 = 0.1 * np.sqrt(np.shape(weight_array_4)[1])

    instance_1 = NoisyLTFArray(weight_array_1, transform, combiner, noise_1, bias=False)
    instance_2 = NoisyLTFArray(weight_array_2, transform, combiner, noise_2, bias=True)
    instance_3 = NoisyLTFArray(weight_array_3, transform, combiner, noise_3, bias=False)
    instance_4 = NoisyLTFArray(weight_array_4, transform, combiner, noise_4, bias=False)

    # set parameters for becker's attack
    priorities_1 = np.array([0.25, 0.25, 0.25, 0.25])
    priorities_2 = np.array([0.4, 0.3, 0.2, 0.1])
    priorities_3 = np.array([.15, .14, .13, .12, .11, .09, .08, .07, .06, .05])
    priorities_4 = np.array([.20, .15, .12, .10, .08, .07, .07, .07, .07, .07])


    test_set = [
        # instance, pop_size, parent_size, priorities
        (instance_1, 8, 4, priorities_1),
        (instance_1, 30, 10, priorities_3),
        (instance_2, 8, 4, priorities_2),
        (instance_2, 30, 10, priorities_4),
        (instance_3, 8, 4, priorities_1),
        (instance_3, 30, 10, priorities_3),
        (instance_4, 8, 4, priorities_2),
        (instance_4, 30, 10, priorities_4)
    ]


p_values = np.array([0.9999, 0.9995, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9])
k_values = np.array([1, 2, 4, 6, 8, 12, 16, 20])

def stability(p, k):
    return ((2*p - 1)**k + 1) / 2

stabilities = np.zeros((8+1, 8+1))
stabilities[0, 1:] = p_values
stabilities[1:, 0] = k_values.T
for i in range(8):
    for j in range(8):
        stabilities[i+1,j+1] = stability(p_values[j], k_values[i])

indices = stabilities > 0.8
copy = np.copy(stabilities)
copy[indices] = 0
stabilities = stabilities - copy



# set pseudo random number generator
prng = np.random.RandomState(0x6C4E)

# build instance of XOR Arbiter PUF to learn
transform = LTFArray.transform_id
combiner = LTFArray.combiner_xor
n = 32
sigma_weight = 1
noisiness = 0.1
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma_weight, noisiness)
k = 4
mu = 0
sigma = sigma_weight
weight_array = LTFArray.normal_weights(n, k, mu, sigma, prng)
instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise, prng)

weight_array_model = LTFArray.normal_weights(n, k, mu, sigma, prng)
model = NoisyLTFArray(weight_array_model, transform, combiner, sigma_noise, prng)

# sample challenges
num = 2**4
challenges = challenges = np.array(list(tools.sample_inputs(n, num, prng)))


def get_particular_accuracies(instance, model, k, challenges):
    challenge_num = np.shape(challenges)[0]
    assert instance.transform == model.transform
    assert instance.combiner == model.combiner
    transform = instance.transform
    combiner = instance.combiner
    accuracies = np.zeros(k)
    for i in range(k):
        model_single_LTFArray = LTFArray(model.weight_array[i, np.newaxis, :], transform, combiner)
        responses_model = model_single_LTFArray.eval(challenges)
        for j in range(k):
            original_single_LTFArray = LTFArray(instance.weight_array[j, np.newaxis, :], transform, combiner)
            responses_original = original_single_LTFArray.eval(challenges)
            accuracy = 0.5 + np.abs(0.5 - (np.count_nonzero(responses_model == responses_original) / challenge_num))
            if accuracy > accuracies[i]:
                accuracies[i] = accuracy
    return accuracies

accuracies = get_particular_accuracies(instance, model, k, challenges)
print('accuracies =', accuracies)



# set parameters as recommended in Hansens CMA-ES p.26
n = 32
lam = np.int16(4 + np.floor(3 * np.log(n)))
mu = np.int16(np.floor(lam / 2))
weights = np.empty(np.int8(mu))
for i in range(mu):
    sum = 0
    for j in range(mu):
        sum += (np.log(mu + 1) - np.log(j+1))
    weights[i] = (np.log(mu + 1) - np.log(i+1)) / sum

mu_w = 1 / np.sum(np.square(weights))
c_sigma = (mu_w + 2) / (n + mu_w + 3)
d_sigma = 1 + 2 * max(0, np.sqrt((mu_w - 1) / (n + 1))) + c_sigma
c_d_sigma = c_sigma / d_sigma
c_c = 4 / (n + 4)
c_mu = (1 / mu_w) * (2 / n + np.sqrt(2) ** 2) + (1 + 1 / mu_w) * min(1, (2 * mu_w - 1) / ((n + 2) ** 2 + mu_w))
c_1 = c_mu / mu_w


print('lam =', lam)
print('mu =', mu)
print('weights\n', weights)
print('mu_w =', mu_w)
print('c_sigma =', c_sigma)
print('d_sigma =', d_sigma)
print('c_d_sigma =', c_d_sigma)
print('c_c =', c_c)
print('c_cov =', c_mu)
