import unittest
import numpy as np
from pypuf import tools
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as becker
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray

n = 6
k = 2
mu = 0
sigma = 1
weight_array = NoisyLTFArray.normal_weights(n, k, mu, sigma)
transform = NoisyLTFArray.transform_id
combiner = NoisyLTFArray.combiner_xor
noisiness = 0.1
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma, noisiness)

instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise)
#print(vars(instance))

pop_size = 5
parent_size = 4
priorities = np.array([.25, .25, .25, .25])

#pop_size = 30
#parent_size = 10
#priorities = np.array([.15, .14, .13, .12, .11, .09, .08, .07, .06, .05])
#priorities = np.array([.20, .15, .12, .10, .08, .07, .07, .07, .07, .07])
bl = becker(instance, pop_size, parent_size, priorities)
#print(vars(becker_learner))

print(bl.mean, '\n', bl.cov_matrix, '\n', bl.pop_size, '\n', bl.step_size)

generation = becker.reproduce(bl.mean, bl.cov_matrix, bl.pop_size, bl.step_size)
print(generation)

class TestLTFArray(unittest.TestCase):

    weight_array_1 = np.array([[0.2, -0.7, 2.4, 0.6, 0.1, -0.9, -1.1, 0.5],
                               [-2.0, 0.7, 1.2, -0.2, 1.3, -0.8, -0.4, 0.9]])
    weight_array_2 = np.array([[1.5, 0.3, -1.2, 0.9, -0.4],
                               [0.1, -2.1, -0.6, 1.4, -0.7]])
    weight_array_3 = LTFArray.normal_weights(32, 4, 1, 3)
    weight_array_4 = LTFArray.normal_weights(64, 8, 0, 1)

    transform = LTFArray.transform_id
    combiner = LTFArray.combiner_xor

    inst_1 = NoisyLTFArray(weight_array_1, transform, combiner,
                          0.1*np.sqrt(np.shape(weight_array_1)[1]), bias=False)
    inst_2 = NoisyLTFArray(weight_array_2, transform, combiner,
                          0.1*np.sqrt(np.shape(weight_array_2)[1]), bias=True)
    inst_3 = NoisyLTFArray(weight_array_3, transform, combiner,
                          0.3 * np.sqrt(np.shape(weight_array_3)[1]), bias=False)
    inst_4 = NoisyLTFArray(weight_array_4, transform, combiner,
                           0.1 * np.sqrt(np.shape(weight_array_4)[1]), bias=False)

    prio_1 = np.array([0.4, 0.3, 0.2, 0.1])
    prio_2 = np.array([.15, .14, .13, .12, .11, .09, .08, .07, .06, .05])

    test_set = [
        # instance, pop_size, parent_size, priorities
        (inst_1, 8, 4, prio_1),
        (inst_1, 30, 10, prio_2),
        (inst_2, 8, 4, prio_1),
        (inst_2, 30, 10, prio_2),
        (inst_3, 8, 4, prio_1),
        (inst_3, 30, 10, prio_2),
        (inst_4, 8, 4, prio_1),
        (inst_4, 30, 10, prio_2)
    ]

    instance = instance  # simulation instance to be modelled
    individuals = np.zeros((instance.n, parent_size))
    # mean, step_size,  pop_size,   parent_size,    priorities, cov_matrix,   path_cm,    path_ss
    # m,    sigma,      lambda,     mu,             w_i,        C,            p_c,        p_sigma
    mean = np.zeros(instance.n)  # mean vector of distribution
    step_size = 1  # distance to next distribution
    pop_size = pop_size  # number of individuals per generation
    parent_size = parent_size  # number of considered individuals
    priorities = priorities  # array of consideration proportion
    cov_matrix = np.identity(instance.n)  # shape of distribution
    path_cm = 0  # cumulated evolution path of covariance matrix
    path_ss = 0  # cumulated evolution path of step size
    # auxiliary constants
    mu_w = 1 / np.sum(np.square(priorities))
    c_mu = mu_w / instance.n ** 2
    d_sigma = 1 + np.sqrt(mu_w / instance.n)
    c_1 = 2 / instance.n ** 2
    c_c = 4 / instance.n
    c_sigma = 4 / instance.n

    # primary methods
    def test_reproduce(self):
        for params in self.test_set:
            shape = np.shape(params[0].weight_array)[1]
            reproduction = becker.reproduce(mean = np.zeros(shape),
                cov_matrix = np.identity(shape), pop_size = params[1],
                step_size = 1)
        matrix = np.array()
        np.testing.assert_array_equal(reproduction, matrix)

    def test_update_mean(self):
        becker.update_mean(old_mean, step_size, parent)

    def test_cumulation_for_cm(self):
        becker.cumulation_for_cm(path_cm, c_c, path_ss, n, mu_w, parent)

    def test_cumulation_for_ss(self):
        becker.cumulation_for_ss(path_ss, c_sigma, mu_w, cov_matrix, parent)

    def test_update_cm(self):
        becker.update_cm(cov_matrix, c_1, c_mu, path_cm, parent_product)

    def test_update_ss(self):
        becker.update_ss(step_size, c_sigma, d_sigma, path_ss)


    # secondary methods
    def test_get_parent(self):
        becker.get_parent(sorted_individuals, parent_size, priorities)

    def test_get_parent_product(self):
        becker.get_cm_mu(sorted_individuals, parent_size, priorities)

    def test_fitness(self):
        becker.fitness(instance, challenges, reliabilities, individuals)

    def test_build_ltf_arrays(self):
        becker.build_ltf_arrays(individuals)

    def test_get_delay_differences(self):
        becker.get_delay_differences(ltf_arrays, pop_size, challenges, challenge_num)

    def test_get_reliabilities(self):
        becker.get_reliabilities(delay_diffs, epsilon)

    def test_get_correlations(self):
        becker.get_correlations(reliabilities, measured_rels)

    def test_sort_individuals(self):
        becker.sort_individuals(individuals, correlations)
