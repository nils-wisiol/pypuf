import numpy as np
from pypuf.learner.evolution_strategies.cma_es import CMA_ES

def f(x, array):
    # returns function value f(x), where array defines polynom f
    dim = len(array)
    sum = 0
    for i in range(dim):
        sum += array[i]*x**i
    return sum

def get_fitness_function(solution):
    # returns a fitness function on a fixed polynom

    def fitness(individuals):
        # returns the differences between candidates and solution of polynom
        number = np.shape(individuals)[0]
        fitness_values = np.zeros(number)
        input_set = np.array([0.1, -0.1, 0.5, -0.5, 1, -1, 2, -2, 3.333, -3.333 , 7.5, -7.5, 10, -10, 20, -20])
        dim = len(input_set)
        for i in range(number):
            diff = 0
            for j in range(dim):
                eval_solution = f(input_set[j], solution)
                eval_individual = f(input_set[j], individuals[i, :])
                diff += np.abs(eval_solution - eval_individual)
            if diff == 0:
                diff = 1
            fitness_values[i] = 1 / diff
        return fitness_values

    return fitness

# test get_fitness_function
solution = np.array([10, 3, 2, -4.5])
individuals = np.array([[2,2,2,2], [0.5, -2, -0.5, 1], [-1.5, 3, 4, -3], [-5, 5, -5, 5]])
fitness_function = get_fitness_function(solution)
fitness_values = fitness_function(individuals)
print('fitness_values\n', fitness_values)

# test sample_mutations
zero_mean = np.array([0, 0, 0, 0])
cov_matrix = np.array([[1, 0, 0.9, 0.5],
                       [0, 1, 0.1, 0],
                       [0.9, 0.1, 1, 0.3],
                       [0.5, 0, 0.3, 1]])
"""cov_matrix = np.array([[0.52923091, -0.0542376, 0.29767901, 0.073488],
                       [-0.0542376, 1.491414, -0.363717, 0.71178],
                       [0.29767901, -0.363717, 1.69502509, 0.574722],
                       [0.073488, 0.71178, 0.574722, 0.6906]])"""
pop_size = 7
mutation_prng = np.random.RandomState(0x1234)
mutations = CMA_ES.sample_mutations(zero_mean, cov_matrix, pop_size, mutation_prng)
print('mutations\n', mutations)

# test reproduce
mean = np.array([1, 2, 3, 4])
pop_size = 2
step_size = 10
mutations = np.array([[0, 1.11, -2, -0.3], [-0.777, 1.5, 0.1234, 1]])
individuals = CMA_ES.reproduce(mean, pop_size, step_size, mutations)
print('individuals\n', individuals)

# test update_mean
mean = np.array([5, 5, 5, 5])
step_size = 0.1
favorite_mutation = np.array([1, 2, 3, 4])
mean = CMA_ES.update_mean(mean, step_size, favorite_mutation)
print('mean\n', mean)

# test cumulation_for_cm
path_cm = np.array([9, 8, 7, 6])
c_c = 0.25
path_ss = np.array([5, 4, 3, 2])
n = 4
mu_w = 10
favorite_mutation = np.array([0.2, 0.4, 0.6, 0.8])
path_cm = CMA_ES.cumulation_for_cm(path_cm, c_c, path_ss, n, mu_w, favorite_mutation)
print('path_cm\n', path_cm)

# test for cuulation_for_ss
path_ss = np.array([3, 3, 3, 3])
c_sigma = 0.25
mu_w = 10
cov_matrix = np.array([[1, 0, 0.9, 0.5],
                       [0, 1, 0.1, 0],
                       [0.9, 0.1, 1, 0.3],
                       [0.5, 0, 0.3, 1]])
favorite = np.array([-0.1, -8, 1, 2])
path_ss = CMA_ES.cumulation_for_ss(path_ss, c_sigma, mu_w, cov_matrix, favorite)
print('path_ss\n', path_ss)

# test for update_cm
cov_matrix = np.array([[1, 0, 0.9, 0.5],
                       [0, 1, 0.1, 0],
                       [0.9, 0.1, 1, 0.3],
                       [0.5, 0, 0.3, 1]])
c_1 = 0.005
c_mu = 0.01
path_cm = np.array([4, 4, 2, 2])
outer_product = np.array([[1, 0, 1.9, -0.5],
                       [0, 1, 0.1, 0],
                       [1.9, 0.1, 1, 0.3],
                       [-0.5, 0, 0.3, 1]])
cov_matrix = CMA_ES.update_cm(cov_matrix, c_1, c_mu, path_cm, outer_product)
print('cov_matrix\n', cov_matrix)

# test for update_ss
step_size = 2
c_d_sigma = 0.15
path_ss = np.array([5, 1, 9, 0])
estimation_multinormal = 5
step_size = CMA_ES.update_ss(step_size, c_d_sigma, path_ss, estimation_multinormal)
print('step_size\n', step_size)

# test for get_favorite_mutations
vector = np.array([[0, 1.11, -2, -0.3],
                   [-0.777, 1.5, 0.1234, 1],
                   [1, 1, 1, 1],
                   [5, 5, 5, 5],
                   [1, 2, 1, 2],
                   [4, 3, 2, 1],
                   [0, 1, -0.5, 3],
                   [-1, -2, -3, -4]])
parent_size = 3
priorities = np.array([0.5, 0.3, 0.2])
favorite_mutations = CMA_ES.get_favorite_mutations(vector, parent_size, priorities)
print('favorite_mutations\n', favorite_mutations)

# test for get_cm_mu
vector = np.array([[0.001, 1.11, -2, -0.3],
                   [-0.777, 1.5, 0.1234, 1],
                   [1, 1, 1, 1],
                   [5, 5, 5, 5],
                   [1, 2, 1, 2],
                   [4, 3, 2, 1],
                   [0, 1, -0.5, 3],
                   [-1, -2, -3, -4]])
parent_size = 3
priorities = np.array([0.34, 0.33, 0.33])
outer_product = CMA_ES.get_mutations_outer_product(vector, parent_size, priorities)
print('outer_product\n', outer_product)

# test modify_eigen_decomposition
matrix = np.array([[1, 0, 0, 0.5],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0.5, 0, 0, 1]])
decomposition = CMA_ES.modify_eigen_decomposition(matrix)
print('decomposition\n', decomposition)

# test is_symmetric
matrix = np.array([[1, 0, 0.9, 0.5],
                   [0, 1, 0.1, 0],
                   [0.9, 0.1, 1, 0.3],
                   [0.5, 0, 0.3, 1]])
symmetry = CMA_ES.is_symmetric(matrix, tol=1e-8)
print('symmetry\n', symmetry)

# test matrix multiplication '@'
vector = np.array([.1, .2, .3, -.4])
product = vector[:, np.newaxis] @ vector[np.newaxis, :]
print('product\n', product)

# test sorting
fitness_values = np.array([0.3, -0.5, 0.8, 0.1])
count_nan = np.count_nonzero(np.isnan(fitness_values))
sorting_indices = np.argsort(fitness_values)
mutations = np.array([1,2,3,4])
sorted_mutations = np.roll(mutations[sorting_indices[::-1]], -count_nan)
print('sorted_mutations\n', sorted_mutations)

# test evolutionary_search
solution = np.array([100, 1000, 0, 5, -7.65, 3, 3, -5, -1.5, 1.05, 0, 1])
fitness_function = get_fitness_function(solution)
n = 12
pop_size = 16
parent_size = 4
weights = np.array([0.4, 0.3, 0.2, 0.1])
step_size_limit = 1 / 2**20
iteration_limit = 2000
prng = np.random.RandomState(0x6803)
cma_es = CMA_ES(fitness_function, n, pop_size, parent_size, weights, step_size_limit, iteration_limit,
                 prng, abortion_function=None)
reached_solution = cma_es.evolutionary_search()
print('reached_solution\n', reached_solution)
#print('CMA-object\n', vars(cma_es))