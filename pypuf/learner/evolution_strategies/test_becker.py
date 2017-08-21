import numpy as np
import itertools as iter
from pypuf import tools
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as Becker
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray


transform = LTFArray.transform_id
combiner = LTFArray.combiner_xor

# test for is_different_LTF checked!
new_LTF = np.array([0.2, -1.1, -0.7, -0.9, -0.49, 1.621, 1.6, 0.77])
different_LTFs = np.array([[-1.2, -0.42, -0.81, 1.62, 0.47, 0.23, -0.37, 0.68],
                           [0.61, -2.5, 0.29, 1.16, -1.83, 0.72, -0.82, 0.775]])
num_of_LTFs = 2
n = 8
num = 17
prng = np.random.RandomState(0x1111)
challenges = np.array(list(tools.sample_inputs(n, num, prng)))
challenge_num = num
is_diff = Becker.is_different_LTF(new_LTF, different_LTFs, num_of_LTFs, challenges, transform, combiner)
print('is_diff\n', is_diff)

# test for get_fitness_function checked!
n = 8
num = 16
prng = prng
challenges = np.array(list(tools.sample_inputs(n, num, prng)))
challenge_num = challenge_num
measured_rels = np.array([1.5, 2.5, 2.5, 2.5, 0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 1.5, 2.5, 0.5, 2.5, 0.5, 1.5])
epsilon = 0.8
from scipy import special as sp
normalize = np.sqrt(2) * sp.gamma((n) / 2) / sp.gamma((n - 1) / 2)
fitness_function = Becker.get_fitness_function(challenges, measured_rels, epsilon, transform, combiner)
print('fitness_function\n', fitness_function)

# test for build_LTFArrays checked!
weight_arrays = np.array([[1,1], [-2,-2]])
built_LTFArrays = Becker.build_LTFArrays(weight_arrays, transform, combiner)
print('built_LTFArrays\n', built_LTFArrays)

# test for get_delay_differences checked!
built_LTFArrays = built_LTFArrays
pop_size = 2
n = 2
num = 4
challenges = np.array(list(tools.sample_inputs(n, num, prng)))
delay_diffs = Becker.get_delay_differences(built_LTFArrays, pop_size, challenges)
print('delay_diffs\n', delay_diffs)

# test for get_reliabilities checked!
delay_diffs = np.array([[0.03, 1.77, -1.48, -2.4],
                        [3.62, 2, 6.2, -1.8],
                        [0.1, -3.2, 2.4, 2.1],
                        [-2.4, 0.6, 4.6, -2.77]])
###epsilons = np.array([0.5, -2.1, 4.2, 1.5])
epsilon = 2.0
reliabilities = Becker.get_reliabilities(delay_diffs, epsilon)
print('reliabilities\n', reliabilities)

# test for get_correlations checked!
reliabilities = np.array([[1, 1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1, 1],
                          [0, 1, 0, 0, 0, 1],
                          [1, 0, 1, 0, 1, 1]])
measured_rels = np.array([2.5, 1.5, 2.5, 0.5, 1.5, 2.5])
correlations = Becker.get_correlations(reliabilities, measured_rels)
print('correlations\n', correlations)

# test for set_pole_of_LTFs checked!
different_LTFs = np.array([[-0.1, 0.15, 0.2], [0.1, 0.15, -0.2]])
challenges = np.array([[1,1,1], [-1,1,-1], [-1,-1,1], [1,-1,-1]])
xor_LTFArray = LTFArray(different_LTFs, transform, combiner)
common_responses = np.array([1, 1, -1, 1])
print('different_LTFs\n', different_LTFs)
polarized_LTFs = Becker.set_pole_of_LTFs(different_LTFs, challenges, common_responses, transform, combiner)
print('polarized_LTFs\n', polarized_LTFs)

# test for is_correlated checked!
responses_new_LTF = np.array([1, 1, 1, 1, 1, -1])
responses_diff_LTFs = np.array([[1, 1, 1, 1, 1, 1],
                                [-1, -1, -1, 1, 1, 1]])
is_same_LTF = Becker.is_correlated(responses_new_LTF, responses_diff_LTFs)
print('is_same_LTF\n', is_same_LTF)

# test for get_common_responses checked!
responses = np.array([[-1, -1, 1, -1, 1, 1, 1],
                      [-1, -1, -1, -1, 1, 1, 1],
                      [1, -1, 1, -1, 1, -1, 1],
                      [-1, -1, 1, -1, 1, 1, 1],
                      [-1, -1, -1, -1, 1, 1, 1]])
common_responses = Becker.get_common_responses(responses)
print('common_responses\n', common_responses)

# test for get_measured_rels checked!
responses = responses
measured_rels = Becker.get_measured_rels(responses)
print('measured_rels\n', measured_rels)

# test for learn
n = 20
sigma_weight = 1
noisiness = 0.05
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma_weight, noisiness)
k = 2
mu = 0
sigma = sigma_weight
weight_array = LTFArray.normal_weights(n, k, mu, sigma, prng)
instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise)
num = 2**13
prng = np.random.RandomState(0xA7E3)
challenges = tools.sample_inputs(n, num, prng)
repetitions = 20
responses_repeated = np.zeros((repetitions, num))
for i in range(repetitions):
    challenges, cs = iter.tee(challenges)
    responses_repeated[i, :] = instance.eval(np.array(list(cs)))
limit_step_size = 1 / 2 ** 10
limit_iteration = 3000
prng = np.random.RandomState(0x1D4)
challenges = np.array(list(challenges))
becker = Becker(k, n, transform, combiner, challenges, responses_repeated, repetitions, limit_step_size,
                limit_iteration, prng)

learned_instance = becker.learn()
responses_model = learned_instance.eval(challenges)
responses_instance = becker.get_common_responses(responses_repeated)
assert len(responses_model) == len(responses_instance)
accuracy = 1 - (num - np.count_nonzero(responses_instance == responses_model)) / num

print('accuracy = ', accuracy)
print('learned_instance', vars(learned_instance))
print('original_instance', vars(instance))
