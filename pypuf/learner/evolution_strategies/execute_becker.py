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

# test set:
# challenge_num, repetitions, noisiness, k, n, limit_step_size, limit_iteration
test_sets = np.array([
    [2 ** 10, 2, 0.1, 1, 32, 1 / 2 ** 8, 2 ** 10],
    [2 ** 10, 2, 0.1, 1, 32, 1 / 2 ** 8, 2 ** 14],
    [2 ** 10, 2, 0.1, 1, 32, 1 / 2 ** 12, 2 ** 10],
    [2 ** 10, 2, 0.1, 1, 32, 1 / 2 ** 12, 2 ** 14],
    [2 ** 10, 2, 0.05, 1, 32, 1 / 2 ** 8, 2 ** 10],
    [2 ** 10, 2, 0.05, 1, 32, 1 / 2 ** 8, 2 ** 14],
    [2 ** 10, 2, 0.05, 1, 32, 1 / 2 ** 12, 2 ** 10],
    [2 ** 10, 2, 0.05, 1, 32, 1 / 2 ** 12, 2 ** 14],
    [2 ** 13, 2, 0.1, 1, 32, 1 / 2 ** 8, 2 ** 10],
    [2 ** 13, 2, 0.1, 1, 32, 1 / 2 ** 8, 2 ** 14],
    [2 ** 13, 2, 0.1, 1, 32, 1 / 2 ** 12, 2 ** 10],
    [2 ** 13, 2, 0.1, 1, 32, 1 / 2 ** 12, 2 ** 14],
    [2 ** 13, 2, 0.05, 1, 32, 1 / 2 ** 8, 2 ** 10],
    [2 ** 13, 2, 0.05, 1, 32, 1 / 2 ** 8, 2 ** 14],
    [2 ** 13, 2, 0.05, 1, 32, 1 / 2 ** 12, 2 ** 10],
    [2 ** 13, 2, 0.05, 1, 32, 1 / 2 ** 12, 2 ** 14]
])

prngs = np.array([0x1111, 0xABC, 0x1234, 0x3D9B])

for set in test_sets:
    for number in prngs:
        # set pseudo random number generator
        prng = np.random.RandomState(number)

        # set test parameters
        challenge_num = int(set[0])
        repetitions = int(set[1])
        noisiness = set[2]
        k = int(set[3])
        n = int(set[4])
        limit_step_size = set[5]
        limit_iteration = set[6]

        # build instance of XOR Arbiter PUF to learn
        transform = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        sigma_weight = 1
        sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma_weight, noisiness)
        mu = 0
        sigma = sigma_weight
        weight_array = LTFArray.normal_weights(n, k, mu, sigma, prng)
        instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise, prng)

        # sample challenges
        challenges = tools.sample_inputs(n, challenge_num, prng)

        # extract responses from instance
        responses_repeated = np.zeros((repetitions, challenge_num))
        for i in range(repetitions):
            challenges, cs = it.tee(challenges)
            responses_repeated[i, :] = instance.eval(np.array(list(cs)))

        # set parameters for CMA-ES
        challenges = np.array(list(challenges))
        becker = Becker(k, n, transform, combiner, challenges, responses_repeated, repetitions, limit_step_size,
                        limit_iteration, prng)

        # learn instance and evaluate solution
        model = becker.learn()
        responses_model = model.eval(challenges)
        responses_instance = becker.get_common_responses(responses_repeated)
        assert len(responses_model) == len(responses_instance)
        accuracy = 1 - tools.approx_dist(instance, model, 2 ** 14)
        accuracy_training = 1 - (challenge_num - np.count_nonzero(responses_instance==responses_model)) / challenge_num
        accuracy_instance = 1 - tools.approx_dist(instance, instance, 2 ** 14)
        iterations = becker.iterations
        abortions = becker.abortions
        challenge_num = challenge_num
        causes = becker.termination_causes
        termination1 = causes[0]
        termination2 = causes[1]
        termination3 = causes[2]
        particular_accuracies = get_particular_accuracies(instance, model, k, challenges)

        # write into csv-file
        import csv
        with open('becker_execs_1.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([accuracy] + [accuracy_training] + [accuracy_instance] + [particular_accuracies]
                            + [iterations] + [abortions] + [challenge_num] + [repetitions] + [noisiness] + [k] + [n]
                            + [limit_step_size] + [limit_iteration] + [termination1] + [termination2] + [termination3]
                            + [number])

        print('learned')

        """
        # print results
        print('accuracy =', accuracy)
        print('accuracy_training =', accuracy_training)
        print('accuracy_instance =', accuracy_instance)
        print('particular_accuracies =', particular_accuracies)
        print('iterations =', iterations)
        print('abortions =', abortions)
        print('challenge_num =', challenge_num)
        print('repetitions =', repetitions)
        print('noisiness =', noisiness)
        print('k =', k)
        print('n =', n)
        print('step_size_limit =', limit_step_size)
        print('iteration_limit =', limit_iteration)
        print('causes =', causes)
        """
print('finished')
