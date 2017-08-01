import numpy as np
from pypuf import tools
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray


n = 10
k = 1
mu = 0
sigma = 1
weight_array = LTFArray.normal_weights(n, k, mu, sigma, random_instance=np.random.RandomState(0x3500))
transform = LTFArray.transform_atf
combiner = LTFArray.combiner_xor
noisiness = 0.1
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma, noisiness)
instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise, random_instance=np.random.RandomState(0x8700))

pop_size = 8
parent_size = 4
priorities = np.array([.25, .25, .25, .25])
#pop_size = 30
#parent_size = 10
#priorities = np.array([.15, .14, .13, .12, .11, .09, .08, .07, .06, .05])
challenge_num = 100
repeat = 10
unreliability = 0.04
precision = 0.6
seed_mutations = 0x6000
seed_inputs = 0x400
becker = learner(instance, pop_size, parent_size, priorities,
                 challenge_num, repeat, unreliability, precision, seed_mutations, seed_inputs)
l = becker.learn()
#print('learned:\n', l.weight_array)


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