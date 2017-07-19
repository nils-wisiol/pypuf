import numpy as np
from pypuf import tools
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray


n = 8
k = 2
mu = 0
sigma = 1
weight_array = NoisyLTFArray.normal_weights(n, k, mu, sigma)
transform = NoisyLTFArray.transform_atf
combiner = NoisyLTFArray.combiner_xor
noisiness = 0.1
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma, noisiness)

instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise)
#print(vars(instance))


pop_size = 30
parent_size = 10
priorities = np.array([.15, .14, .13, .12, .11, .09, .08, .07, .06, .05])
challenge_num = 100
repeat = 5
unreliability = 0.04
precision = 0.6
becker = learner(instance, pop_size, parent_size, priorities,
                 challenge_num, repeat, unreliability, precision)
l = becker.learn()
print('learned:\n', l.weight_array)

pop_size = 5
parent_size = 4
priorities = np.array([.25, .25, .25, .25])

"""

different_LTFs2 = np.array([-0.5, -0.5, -1, 1, -2, 1])
different_LTFs3 = np.array([0, 1, 2, -0.5, 0.5, 3])
new_LTF = np.array([0.5, 0.5, 0.5, -0.5, 0.5, 0.5])

ltf2 = LTFArray(different_LTFs2, transform, combiner, sigma_noise)
ltf3 = LTFArray(different_LTFs3, transform, combiner, sigma_noise)
ltf_new = LTFArray(new_LTF, transform, combiner, sigma_noise)

challenges = tools.sample_inputs(6, 10)



different_LTFs1 = np.array([[0.5, -1, -0.5211, 1, 1.5, -1.5, -0.11, 1.4], [-1.22, 0.6, 0.5, 0.666, -0.5211, 1, -1.5, -1.54]])
ltfs = learner.build_LTFArrays(different_LTFs1)
challenges = np.array(list(tools.sample_inputs(9, 100)))
for i in ltfs:
    print(i.val(challenges))
"""