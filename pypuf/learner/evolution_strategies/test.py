import itertools
import numpy as np
from pypuf import tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as learner

cnt = np.count_nonzero(np.isnan(a))
c = np.roll(b, -cnt)

a = np.array([  1.,  -1.,   np.nan,  0.,  np.nan], dtype=np.float32)
print(a)
cnt = np.count_nonzero(np.isnan(a))
print('cnt: ', cnt)
ind = np.argsort(a)
b = a[ind[::-1]]
print('b\n', b)
c = np.roll(b, -cnt)
print('c: ', c)


sa = np.argsort(a)[::-1]
print(sa)
asa = np.roll(sa,-np.count_nonzero(np.isnan(a)))
print(asa)

"""
def modify_eigen_decomposition(matrix):
    # returns modified eigen-decomposition (B * D^(-1) * B^T) of matrix A = (B * D^2 * B^T) (corresponds to C^(-1/2))
    eigen_values, eigen_vectors = np.linalg.eigh(matrix)
    diagonal = np.sqrt(np.diag(eigen_values))
    return eigen_vectors[:, np.newaxis] @ (1 / diagonal) @ eigen_vectors[np.newaxis, :]
"""


def modify_eigen_decomposition(matrix):
    # returns modified eigen-decomposition (B * D^(-1) * B^T) of matrix A = (B * D^2 * B^T) (corresponds to C^(-1/2))
    eigen_values, eigen_vectors = np.linalg.eigh(matrix)
    diagonal = 1 / np.diag(np.sqrt(np.diag(eigen_values)))
    return eigen_vectors @ np.diag(diagonal) @ eigen_vectors.T

"""
def modify_eigen_decomposition(matrix):
    # returns modified eigen-decomposition (B * D^(-1) * B^T) of matrix A = (B * D^2 * B^T) (corresponds to C^(-1/2))
    eigen_values, eigen_vectors = np.linalg.eigh(matrix)
    print('eigen_values\n', eigen_values, 'eigen_vectors\n', eigen_vectors)
    diagonal = np.sqrt(np.diag(eigen_values))
    print('diagonal\n', diagonal)
    print('eigen_vectors @ (1/diagonal)\n', eigen_vectors @ diagonal**(-1), '@ eigen_vectors.T\n', eigen_vectors @ (1/diagonal) @ eigen_vectors.T)
    return eigen_vectors @ (1/diagonal) @ eigen_vectors.T
"""

a = np.array([[2,0], [0,2]])
#b = 1/a
#print('bbbbbbbb', b)

mat = np.array([[10,2,3], [2,10,6], [3,6,10]])
dec = modify_eigen_decomposition(mat)
print('dec\n', dec)

a = np.array([1,2,3])
b = np.array([[1,1,1], [1,1,1], [1,1,1]])
c = a[:, np.newaxis] @ a[np.newaxis, :]
print('c\n', c)
c = a@b
print('c\n', c)

"""
def modify_eigen_decomposition(matrix):
    # returns modified eigen-decomposition (B * D^(-1) * B^T) of matrix A = (B * D^2 * B^T) (corresponds to C^(-1/2))
    eigen_values, eigen_vectors = np.linalg.eigh(matrix)
    diagonal = np.sqrt(np.diag(eigen_values))
    diagonal_inverse = np.linalg.inv(diagonal)
    return eigen_vectors @ diagonal_inverse @ eigen_vectors.T
"""

def cumulation_for_ss(path_ss, c_sigma, mu_w, cov_matrix, parent):
    # returns cumulated evolution path of step-size (corresponds to p_sigma)
    cm_eigen_dec = learner.modify_eigen_decomposition(cov_matrix)
    return (1 - c_sigma) * path_ss + np.sqrt(1 - (1 - c_sigma) ** 2) * np.sqrt(mu_w) * cm_eigen_dec @ parent


path_ss = np.array([-0.34485763, 1.11920498, -0.49091369, -2.78254796, 0.02105892, -5.12057504, 2.60987016, 3.02267444,
                    -0.19326213, 1.52338899, 0.80677901])
c_sigma = 0.36363636363636365
mu_w = 4.0
cov_matrix = np.array([[0.61842989, 0.42294264, 0.03588551, 0.24631514, 0.24380282, 0.52452589, 0.13642069, 0.1507718, 0.44697789, 0.12312227, 0.9503228],
                       [0.20485916, 0.93282122, 0.1399231, 0.20006003, 0.22217405, 0.55842256, 0.153686, 0.19383252, 0.49561711, 0.16835053, 0.98860887],
                       [-0.00146148, 0.3206596, 0.61310816, 0.22702649, 0.21207061, 0.46475151, 0.19473242, 0.14380727, 0.40502616, 0.15206631, 0.67928963],
                       [0.09919773, 0.2710261, 0.11725606, 0.755224, 0.25845303, 0.55072851, 0.17860856, 0.17158378, 0.43796015, 0.16237649, 0.82313866],
                       [0.15376438, 0.35021909, 0.15937915, 0.31553199, 0.78021397, 0.5565266, 0.14505443, 0.22198906, 0.44662784, 0.09861747, 0.87262167],
                       [0.11749206, 0.36947221, 0.09506466, 0.29081208, 0.23953121, 1.23868475, 0.26000033, 0.12934855, 0.53280087, 0.1967795, 0.93000659],
                       [0.08627627, 0.32162507, 0.18193499, 0.27558156, 0.18494846, 0.61688975, 0.650942, 0.15449631, 0.43014057, 0.08748107, 0.73781146],
                       [0.10514125, 0.36628546, 0.13552371, 0.27307065, 0.26639696, 0.49075183, 0.15901018, 0.61188589, 0.37888473, 0.12812317, 0.81323803],
                       [0.16221377, 0.42893648, 0.15760903, 0.30031345, 0.25190217, 0.65507059, 0.19552087, 0.13975116, 1.0348986, 0.22866155, 1.00660005],
                       [0.07411718, 0.33742892, 0.14040821, 0.26048881, 0.13965083, 0.55480825, 0.0886204, 0.12474863, 0.46442058, 0.66442489, 0.80121933],
                       [0.38109456, 0.63746411, 0.14740838, 0.40102783, 0.39343187, 0.76781219, 0.21872763, 0.28964033, 0.72213593, 0.28099618, 1.9619393]])
parent_mutations = np.array([0.59672195, 0.46136144, -0.05686392, 0.00615447, -0.21924931, 0.80377518, -0.0216225, 0.2814175, 0.39674276, 0.24782712, 0.55362269])

eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
print('eigen_values\n', eigen_values, '\neigen_vectors\n', eigen_vectors)

diag = np.diag(eigen_values)
print('diag\n', diag)
diagonal = np.sqrt(np.diag(eigen_values))
print('diagonal\n', diagonal)

print('decomposition\n', learner.modify_eigen_decomposition(cov_matrix))

new_path = cumulation_for_ss(path_ss, c_sigma, mu_w, cov_matrix, parent_mutations)
print('new_path\n', new_path)

path_ss = np.array([2,2,2,2])
norm = np.linalg.norm(path_ss)
print('norm', norm)

seed_mutations = 0x5000
mutation_prng = np.random.RandomState(seed=seed_mutations)
mean = np.array([0,0,0,0])
cov_matrix = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
pop_size = 5
mutations = mutation_prng.multivariate_normal(np.zeros(np.shape(mean)), cov_matrix, pop_size)
print('mutations', mutations)



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

"""
@staticmethod
def fitness(challenges, challenge_num, measured_rels, individuals):
    # returns individuals sorted by their correlation coefficient as fitness
    pop_size = np.shape(individuals)[0]
    built_LTFArrays = __class__.build_LTFArrays(individuals[:, :-1])
    delay_diffs = __class__.get_delay_differences(built_LTFArrays, pop_size, challenges, challenge_num)
    epsilons = individuals[:, -1]
    reliabilities = __class__.get_reliabilities(delay_diffs, epsilons)
    correlations = __class__.get_correlations(reliabilities, measured_rels)
    correlations = [-.3 if x != x else x for x in correlations]
    return correlations
"""
