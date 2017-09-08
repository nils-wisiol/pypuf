import numpy as np
import itertools as it
from pypuf import tools
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as Becker
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray


def get_particular_accuracies(instance, model, k, challenges):
    challenge_num = np.shape(challenges)[0]
    assert instance.transform == model.transform
    assert instance.combiner == model.combiner
    transform = instance.transform
    combiner = instance.combiner
    accuracies = np.zeros(k)
    for i in range(k):
        model_single_LTFArray = LTFArray(model.weight_array[i, np.newaxis, :], transform, combiner)
        responses_model = model_single_LTFArray.eval(challenges)
        for j in range(k):
            original_single_LTFArray = LTFArray(instance.weight_array[j, np.newaxis, :], transform, combiner)
            responses_original = original_single_LTFArray.eval(challenges)
            accuracy = 0.5 + np.abs(0.5 - (np.count_nonzero(responses_model == responses_original) / challenge_num))
            if accuracy > accuracies[i]:
                accuracies[i] = accuracy
    return accuracies

# set path
path = 'becker_execs.csv'

# set parameters
pop_sizes = np.array([20])
challenge_nums = np.array([2**16])
repetitions = np.array([2**3])
noisinesses = np.array([1/2**3])
ks = np.array([4])
ns = np.array([32])
limits_s = np.array([1/2**12])
limits_i = np.array([2**12])

# set pseudo random number generator
prng = np.random.RandomState(0xA1A7)

# execute with parameters as set above
for pop_size in pop_sizes:
    for challenge_num in challenge_nums:
        for repetition in repetitions:
            for noisiness in noisinesses:
                for k in ks:
                    for n in ns:
                        for limit_s in limits_s:
                            for limit_i in limits_i:

                                # set test parameters
                                challenge_num = int(challenge_num)
                                repetition = int(repetition)
                                noisiness = noisiness
                                k = int(k)
                                n = int(n)
                                limit_step_size = limit_s
                                limit_iteration = limit_i

                                # build instance of XOR Arbiter PUF to learn
                                transform = LTFArray.transform_id
                                combiner = LTFArray.combiner_xor
                                sigma_weight = 1
                                sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma_weight,
                                                                                            noisiness)
                                mu = 0
                                sigma = sigma_weight
                                weight_array = LTFArray.normal_weights(n, k, mu, sigma, prng)
                                instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise, prng)

                                # sample challenges
                                challenges = tools.sample_inputs(n, challenge_num, prng)

                                # extract responses from instance
                                responses_repeated = np.zeros((repetition, challenge_num))
                                for i in range(repetition):
                                    challenges, cs = it.tee(challenges)
                                    responses_repeated[i, :] = instance.eval(np.array(list(cs)))

                                # set parameters for CMA-ES
                                challenges = np.array(list(challenges))
                                becker = Becker(k, n, transform, combiner, challenges, responses_repeated,
                                                repetition, limit_step_size, limit_iteration, prng)
                                becker.pop_size = pop_size

                                # learn instance and evaluate solution
                                model = becker.learn()
                                responses_model = model.eval(challenges)
                                responses_instance = becker.get_common_responses(responses_repeated)
                                assert len(responses_model) == len(responses_instance)
                                accuracy = 1 - tools.approx_dist(instance, model, 2 ** 14)
                                accuracy_training = 1 - (challenge_num - np.count_nonzero(
                                    responses_instance == responses_model)) / challenge_num
                                accuracy_instance = 1 - tools.approx_dist(instance, instance, 2 ** 14)
                                iterations = becker.iterations
                                abortions = becker.abortions
                                challenge_num = challenge_num
                                causes = becker.termination_causes
                                termination1 = causes[0]
                                termination2 = causes[1]
                                termination3 = causes[2]
                                particular_accuracies = get_particular_accuracies(instance, model, k, challenges)

                                # write results into csv-file
                                import csv

                                with open(path, 'a', newline='') as csvfile:
                                    writer = csv.writer(csvfile, delimiter=';', quotechar='|',
                                                        quoting=csv.QUOTE_MINIMAL)
                                    writer.writerow(
                                        [accuracy] + [accuracy_training] + [accuracy_instance]
                                        + [particular_accuracies] + [iterations] + [abortions] + [challenge_num]
                                        + [repetition] + [noisiness] + [k] + [n] + [limit_step_size]
                                        + [limit_iteration] + [termination1] + [termination2] + [termination3]
                                        + [pop_size])

                                print('...learned...')

                                # print results
                                print('accuracy =', accuracy)
                                print('accuracy_training =', accuracy_training)
                                print('accuracy_instance =', accuracy_instance)
                                print('particular_accuracies =', particular_accuracies)
                                print('iterations =', iterations)
                                print('abortions =', abortions)
                                print('challenge_num =', challenge_num)
                                print('repetitions =', repetition)
                                print('noisiness =', noisiness)
                                print('k =', k)
                                print('n =', n)
                                print('pop_size =', pop_size)
                                print('step_size_limit =', limit_step_size)
                                print('iteration_limit =', limit_iteration)
                                print('causes =', causes)

print('___finished___')
