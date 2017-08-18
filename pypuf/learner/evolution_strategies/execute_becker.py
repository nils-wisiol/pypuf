import numpy as np
import itertools as iter
from pypuf import tools
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as Becker
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray

# set pseudo random number generator
prng = np.random.RandomState(0x37BF)

# build instance of XOR Arbiter PUF to learn
transform = LTFArray.transform_id
combiner = LTFArray.combiner_xor
n = 20
sigma_weight = 1
noisiness = 0.2
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma_weight, noisiness)
k = 1
mu = 0
sigma = sigma_weight
weight_array = LTFArray.normal_weights(n, k, mu, sigma, prng)
instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise, random_instance=prng)

# sample challenges
num = 2**10
challenges = tools.sample_inputs(n, num, prng)

# extract responses from instance
repeat = 5
responses_repeated = np.zeros((repeat, num))
for i in range(repeat):
    challenges, cs = iter.tee(challenges)
    responses_repeated[i, :] = instance.eval(np.array(list(cs)))

# set parameters for CMA-ES
step_size_limit = 1 / 2**9
iteration_limit = 2000
pop_size = 30
parent_size = 9
#priorities = np.array([.20, .15, .12, .10, .08, .07, .07, .07, .07, .07])
#priorities = np.array([.16, .14, .13, .12, .11, .1, .09, .08, .07])
priorities = np.array([.12, .11, .11, .11, .11, .11, .11, .11, .11])
#priorities = np.array([.13, .13, .13, .13, .12, .12, .12, .12])
challenges = np.array(list(challenges))
becker = Becker(k, n, transform, combiner, challenges, responses_repeated, repeat, step_size_limit,
                 iteration_limit, pop_size, parent_size, priorities, prng)

# learn instance and evaluate solution
learned_instance = becker.learn()
responses_model = learned_instance.eval(challenges)
responses_instance = becker.get_common_responses(responses_repeated)
assert len(responses_model) == len(responses_instance)
accuracy = 1 - (num - np.count_nonzero(responses_instance == responses_model)) / num

# print results
print('accuracy = ', accuracy)
print('learned_instance', vars(learned_instance))
print('original_instance', vars(instance))

# write into csv-file
import csv
with open('becker_execs.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #writer.writerow([accuracy] + [iterations] + [attempts] + [challenge_num] + [n] + [k])