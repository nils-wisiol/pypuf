import unittest
import numpy as np
from pypuf import tools
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as Becker
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

len = np.shape(stabilities)[0]
for i in range(len):
    print(np.array_str(stabilities[i, :]))
#print(stabilities)