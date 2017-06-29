from pypuf.learner.evolution_strategies.becker import reliability_based_CMA_ES as becker
from pypuf.simulation.arbiter_based.ltfarray import NoisyLTFArray
from pypuf import tools
import numpy as np

n = 8
k = 4
mu = 0
sigma = 1
weight_array = NoisyLTFArray.normal_weights(n, k, mu, sigma)
transform = NoisyLTFArray.transform_id
combiner = NoisyLTFArray.combiner_xor
noisiness = 0.1
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma, noisiness)


instance = NoisyLTFArray.__init__(weight_array, transform, combiner, sigma_noise, noise_seed, bias)
pop_size = 30
parent_size = 9
priorities = np.array([.20, .17, .15, .13, .11, .9, .7, .5, .3])
becker_learner = becker(instance, pop_size, parent_size, priorities,
                 mu_weight=0, sigma_weight=1)


generation = becker.reproduce(mean, cov_matrix, pop_size, step_size)