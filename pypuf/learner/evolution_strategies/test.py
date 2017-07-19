import itertools
import numpy as np
from pypuf import tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as learner


def reproduce(mean, cov_matrix, pop_size, step_size):
    # returns a new generation of individuals as 2D array (pop_size, n)
    mutations = np.random.multivariate_normal(np.zeros(np.shape(mean)), cov_matrix, pop_size)
    duplicated_mean = np.tile(mean, (pop_size, 1))
    return duplicated_mean + (step_size * mutations)

mean = np.array([100,10,0,0])
cov_matrix = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
pop_size = 6
step_size = 1

repr = reproduce(mean, cov_matrix, pop_size, step_size)
print('repr:\n', repr, np.shape(repr))
print('cov_matrix:\n', cov_matrix)



"""
def get_cm_mu(sorted_individuals, parent_size, priorities):
    # returns the weighted sum of the fittest individuals
    cm_mu = np.zeros((parent_size, parent_size))
    sorted_col = np.copy(sorted_individuals)[:, :, np.newaxis]
    sorted_row = np.copy(sorted_individuals)[:, np.newaxis, :]
    for i in range(parent_size):
        cm_mu += priorities[i] * sorted_individuals[i, :] @ sorted_individuals[i, :]
    return cm_mu

def get_cm_mu(sorted_individuals, parent_size, priorities):
    # returns the weighted sum of the fittest individuals
    cm_mu = np.zeros((np.shape(sorted_individuals)[1], np.shape(sorted_individuals)[1]))
    for i in range(parent_size):
        cm_mu += priorities[i] * sorted_individuals[:, :, np.newaxis] @ sorted_individuals[:, np.newaxis, :]
    return cm_mu


bla = np.array([[-1, 0, 2, 1], [1, 9, 2, 1]])
print(np.shape(bla[0, np.newaxis, :]))
cm_mu = np.zeros((np.shape(bla)[1], np.shape(bla)[1]))
for i in range(2):
    cm_mu += 0.5 * bla[i, :, np.newaxis] @ bla[i, np.newaxis, :]


#print('get_cmu......\n', get_cm_mu(bla, 2, np.array([0.6, 0.4])))
ble = np.copy(bla)[:, :, np.newaxis]
bli = np.copy(bla)[:, np.newaxis, :]
print('ble\n', ble, '\nbli\n', bli)
print('ble @ bli\n', ble[0, :] @ bli[0, :] + ble[1, :] @ bli[1, :])
print('bla---\n', bla[0, :, np.newaxis] @ bla[1, np.newaxis, :])
blu = np.array([[-1], [0], [2], [1]])
#print('bla---\n', blu @ bla)



sorted_individuals = np.array([[[-0.1,0.33,1,-0.3]], [[0.567,-0.8,-1.1,0.9]], [[0,0.32,1,-1]]])
parent_size = 2
priorities = np.array([0.6, 0.4])
#cm_mu = get_cm_mu(sorted_individuals, parent_size, priorities)
#print('cm_mu', cm_mu)



n = 4
k = 1
transform = LTFArray.transform_id
combiner = LTFArray.combiner_xor

weight_array1 = LTFArray.normal_weights(n, k, mu=0, sigma=1)
instance1 = LTFArray(weight_array1, transform, combiner, bias=False)

weight_array2 = LTFArray.normal_weights(n, k, mu=0, sigma=1)
instance2 = LTFArray(weight_array2, transform, combiner, bias=False)
"""



"""
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, 1, 0.01)
print('sigma_noise\n', sigma_noise)
instance = NoisyLTFArray(weight_array1, transform, combiner, sigma_noise)
num = 8
challenge_num = num
repeat = 5
challenges = np.array(list(tools.sample_inputs(n, num)))
print('challenges\n', challenges)
print('measured_rels', measure_rels(instance, challenges, challenge_num, repeat))

n = 4
num = 8
challenges = np.array(list(tools.sample_inputs(n, num)))
challenge_num = num
measured_rels = np.array([0.5, 1, 2, 2.5, 2.5, 1, 2.5, 2.5])
individuals = np.array([[-1, 0.7, 0, -0.33], [0.19, -2.3, 1.9, 0.1]])
"""
def fitness(challenges, challenge_num, measured_rels, individuals):
    # returns individuals sorted by their fitness
    pop_size = np.shape(individuals)[0]
    built_LTFs = learner.build_LTFArrays(individuals[:, :-1])
    delay_diffs = learner.get_delay_differences(built_LTFs, pop_size, challenges, challenge_num)
    epsilons = individuals[:, -1]
    reliabilities = learner.get_reliabilities(delay_diffs, epsilons)
    correlations = learner.get_correlations(reliabilities, measured_rels)
    return correlations
#print(fitness(challenges, challenge_num, measured_rels, individuals))

"""
mean = np.array([-1, 0, 1, 2.5])
step_size = -1.5
parent = np.array([0.5, -2, 1, -0.5])
path_cm = np.array([1,2,3,4])
c_c = 0.2
path_ss = np.array([[0.5,-0.5,0,1]])
n = 4
mu_w = 4
c_sigma = 0.2
cov_matrix = np.array([[1,0.5,0,0], [0.5,1,0,0.5], [0,0,1,0.2], [0,0.5,0.2,1]])
cm_eigen_dec = learner.modify_eigen_decomposition(cov_matrix)
print('cm_eigen_dec\n', cm_eigen_dec)
print('update_mean\n', update_mean(mean, step_size, parent))
print('cumulation_cm\n', cumulation_for_cm(path_cm, c_c, path_ss, n, mu_w, parent))
print('cumulation_ss\n', cumulation_for_ss(path_ss, c_sigma, mu_w, cov_matrix, parent))

def update_cm(cov_matrix, c_1, c_mu, path_cm, cm_mu):
    # returns covariance matrix of a new population (pop_size, pop_size)
    return (1 - c_1 - c_mu) * cov_matrix + c_1 * path_cm * path_cm.T + c_mu * cm_mu

c_1 = 0.25
c_mu = 0.3
cm_mu = cov_matrix
print('update_cm\n', update_cm(cov_matrix, c_1, c_mu, path_cm, cm_mu))

def update_ss(step_size, c_sigma, d_sigma, path_ss):
    # returns step-size of a new population
    factor = np.exp((c_sigma / d_sigma) * ((np.linalg.norm(path_ss) / np.sqrt(np.shape(path_ss)[1])) - 1))
    return step_size * factor

d_sigma = 1.2
print('update_ss\n', update_ss(step_size, c_sigma, d_sigma, path_ss))


def get_cm_mu(sorted_individuals, parent_size, priorities):
    # returns the weighted sum of the fittest individuals
    cm_mu = np.zeros((parent_size, parent_size))
    sorted_col = np.copy(sorted_individuals)[:, :, np.newaxis]
    sorted_row = np.copy(sorted_individuals)[:, np.newaxis, :]
    for i in range(parent_size):
        cm_mu += priorities[i] * sorted_col[i, :] @ sorted_row[i, :]
    return cm_mu

bla = np.array([[-1, 0, 2, 1], [1, 9, 2, 1]])
ble = np.copy(bla)[:, :, np.newaxis]
bli = np.copy(bla)[:, np.newaxis, :]
print('ble\n', ble, '\nbli\n', bli)
print('ble @ bli\n', ble[0, :] @ bli[0, :] + ble[1, :] @ bli[1, :])
print('bla---\n', bla[0, :, np.newaxis] @ bla[1, np.newaxis, :])
blu = np.array([[-1], [0], [2], [1]])
#print('bla---\n', blu @ bla)

sorted_individuals = np.array([[0.5, -1, 2, -0.8], [0.5, 0.5, 1, -0.5], [3, -0.5, -1, -1], [2, 1, 3, 0]])
print('newaxis\n', sorted_individuals[0, :, np.newaxis])
parent_size = 3
priorities = np.array([0.5, 0.25, 0.25])
print('get_cm_mu\n', get_cm_mu(sorted_individuals, parent_size, priorities))
"""
########################################################################################################################
@staticmethod
def learn_epsilon(instance, unreliability):
    # returns epsilon on the assumption:
    #   98% of challenges are reliable on a single PUF,
    #   while there are 1/2 * ((96%)^k + 1) on k XORed PUFs
    n = instance.n
    k = instance.k
    num = 1000
    repeat = 100
    quantile = ((1-2*unreliability)**k + 1) / 2
    cut = num - int(num * quantile)
    challenges = tools.sample_inputs(n, num)
    ratios = np.zeros(num)
    responses = np.zeros((num, repeat))
    for i, challenge in enumerate(challenges):
        for j in range(repeat):
            responses[i, j] = instance.eval(challenge)
        ratio = np.sum(responses[i, :]) / repeat
        ratios[i] = ratio if ratio <= 0.5 else (1 - ratio)
    ratios = np.sort(ratios)
    return np.sum(ratios[cut-3 : cut+3]) / 10