import numpy as np
from scipy import special as sp, linalg as li
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.evolution_strategies.cma_es import CMA_ES

class Reliability_based_CMA_ES():

    # recommended properties of parameters:
    #   pop_size=30
    #   parent_size=10 (>= 0.3*pop_size)
    #   priorities: linearly low decreasing
    def __init__(self, k, n, transform, combiner, challenges, responses_repeated, repeat, step_size_limit,
                 iteration_limit, pop_size, parent_size, priorities, prng=np.random.RandomState()):
        self.k = k                                          # number of XORed LTFs
        self.n = n                                          # length of LTFs
        self.transform = transform                          # function for modifying challenges
        self.combiner = combiner                            # function for combining the particular LTFArrays
        self.challenges = challenges                        # set of challenges applied on instance to learn
        self.responses_repeated = responses_repeated        # responses of repeatedly evaluated challenges on instance
        self.repeat = repeat                                # frequency of same repeated challenge
        self.different_LTFs = np.zeros((self.k, self.n))    # all currently learned LTFs
        self.num_of_LTFs = 0                                # number of different learned LTFs
        self.measured_rels = self.get_measured_rels(self.responses_repeated)    # measured reliabilities (instance)
        # parameters for CMA_ES
        self.prng = prng                                    # pseudo random number generator (CMA_ES)
        self.step_size_limit = step_size_limit              # intended scale of step-size to achieve (CMA_ES)
        self.iteration_limit = iteration_limit              # maximum number of iterations within learning (CMA_ES)
        self.pop_size = pop_size                            # number of models sampled each iteration (CMA_ES)
        self.parent_size = parent_size                      # number of considered models each iteration (CMA_ES)
        self.priorities = priorities                        # array of consideration proportions (CMA_ES)

    def learn(self):
        # this is the general learning method
        # returns an XOR-LTFArray with nearly the same behavior as learned instance
        epsilon = self.n / 10
        fitness_function = self.get_fitness_function(self.challenges, self.measured_rels, epsilon,
                                                     self.transform, self.combiner)
        normalize = np.sqrt(2) * sp.gamma((self.n) / 2) / sp.gamma((self.n - 1) / 2)
        while self.num_of_LTFs < self.k:
            abortion_function = self.get_abortion_function(self.different_LTFs, self.num_of_LTFs, self.challenges,
                                                           self.transform, self.combiner)
            cma_es = CMA_ES(fitness_function, self.n, self.pop_size, self.parent_size, self.priorities,
                            self.step_size_limit, self.iteration_limit, self.prng, abortion_function)
            new_LTF = cma_es.evolutionary_search()
            if self.is_different_LTF(new_LTF, self.different_LTFs, self.num_of_LTFs, self.challenges, self.transform,
                                     self.combiner):
                self.different_LTFs[self.num_of_LTFs] = new_LTF * normalize / li.norm(new_LTF)  # normalize weights
                self.num_of_LTFs += 1
        common_responses = self.get_common_responses(self.responses_repeated)
        self.different_LTFs = self.set_pole_of_LTFs(self.different_LTFs, self.challenges, common_responses,
                                                    self.transform, self.combiner)
        return LTFArray(self.different_LTFs, self.transform, self.combiner, bias=False)

    @staticmethod
    def get_fitness_function(challenges, measured_rels, epsilon, transform, combiner):
        # returns a fitness function on a fixed set of challenges and corresponding reliabilities
        becker = __class__

        def fitness(individuals):
            # returns individuals sorted by their correlation coefficient as fitness
            pop_size = np.shape(individuals)[0]
            built_LTFArrays = becker.build_LTFArrays(individuals, transform, combiner)
            delay_diffs = becker.get_delay_differences(built_LTFArrays, pop_size, challenges)
            reliabilities = becker.get_reliabilities(delay_diffs, epsilon)
            correlations = becker.get_correlations(reliabilities, measured_rels)
            return correlations

        return fitness

    @staticmethod
    def get_abortion_function(different_LTFs, num_of_LTFs, challenges, transform, combiner):
        # returns an abortion function on a fixed set of challenges and LTFs
        becker = __class__
        weight_arrays = different_LTFs[:num_of_LTFs, :]
        different_LTFArrays = becker.build_LTFArrays(weight_arrays, transform, combiner)
        responses_diff_LTFs = np.zeros((num_of_LTFs, np.shape(challenges)[0]))
        for i, current_LTF in enumerate(different_LTFArrays):
            responses_diff_LTFs[i, :] = current_LTF.eval(challenges)

        def abortion_function(new_LTF):
            if num_of_LTFs == 0:
                return False
            new_LTFArray = LTFArray(new_LTF[np.newaxis, :], transform, combiner)
            responses_new_LTF = new_LTFArray.eval(challenges)
            return becker.is_correlated(responses_new_LTF, responses_diff_LTFs)

        return abortion_function

    @staticmethod
    def is_different_LTF(new_LTF, different_LTFs, num_of_LTFs, challenges, transform, combiner):
        # returns True, if new_LTF is different from previously learned LTFs
        if num_of_LTFs == 0:
            return True
        weight_arrays = different_LTFs[:num_of_LTFs, :]
        new_LTFArray = LTFArray(new_LTF[np.newaxis, :], transform, combiner)
        different_LTFArrays = __class__.build_LTFArrays(weight_arrays, transform, combiner)
        responses_new_LTF = new_LTFArray.eval(challenges)
        responses_diff_LTFs = np.zeros((num_of_LTFs, np.shape(challenges)[0]))
        for i, current_LTF in enumerate(different_LTFArrays):
            responses_diff_LTFs[i, :] = current_LTF.eval(challenges)
        return not __class__.is_correlated(responses_new_LTF, responses_diff_LTFs)

    @staticmethod
    def set_pole_of_LTFs(different_LTFs, challenges, common_responses, transform, combiner):
        # returns the correctly polarized XOR-LTFArray
        xor_LTFArray = LTFArray(different_LTFs, transform, combiner)
        responses_model = xor_LTFArray.eval(challenges)
        difference = np.sum(np.abs(common_responses - responses_model))
        polarized_LTFs = different_LTFs
        if difference > np.shape(challenges)[0]:
            polarized_LTFs[0, :] *= -1
        return polarized_LTFs


    # methods for calculating fitness
    @staticmethod
    def build_LTFArrays(weight_arrays, transform, combiner):
        # returns iterator over ltf_arrays created out of every individual
        pop_size = np.shape(weight_arrays)[0]
        for i in range(pop_size):
            yield LTFArray(weight_arrays[i, np.newaxis, :], transform, combiner, bias=False)

    @staticmethod
    def get_delay_differences(built_LTFArrays, pop_size, challenges):
        # returns 2D array of delay differences for all challenges on every individual
        delay_diffs = np.empty((pop_size, np.shape(challenges)[0]))
        for i, built_LTFArray in enumerate(built_LTFArrays):
            delay_diffs[i, :] = built_LTFArray.val(challenges)
        return delay_diffs

    @staticmethod
    def get_reliabilities(delay_diffs, epsilon):
        # returns 2D array of reliabilities for all challenges on every individual
        reliabilities = np.zeros(np.shape(delay_diffs))
        for i in range(np.shape(reliabilities)[0]):
            indices_of_reliable = np.abs(delay_diffs[i, :]) > epsilon
            reliabilities[i, indices_of_reliable] = 1
        return reliabilities

    @staticmethod
    def get_correlations(reliabilities, measured_rels):
        # returns array of pearson correlation coefficients between reliability array of individual
        #   and instance for all individuals
        pop_size = np.shape(reliabilities)[0]
        correlations = np.zeros(pop_size)
        ones = np.full(np.shape(reliabilities)[1], 1)
        for i in range(pop_size):
            if np.any(reliabilities[i, :]) and not np.array_equal(reliabilities[i, :], ones):
                correlations[i] = np.corrcoef(reliabilities[i, :], measured_rels)[0, 1]
            else:
                correlations[i] = -1
        return correlations


    # helping methods
    @staticmethod
    def is_correlated(responses_new_LTF, responses_diff_LTFs):
        # returns True, if 2 response arrays are more than 75% equal (Hamming distance)
        num_of_LTFs, challenge_num = np.shape(responses_diff_LTFs)
        for i in range(num_of_LTFs):
            differences = np.sum(np.abs(responses_new_LTF[:] - responses_diff_LTFs[i, :])) / 2
            if differences < 0.25*challenge_num or differences > 0.75*challenge_num:
                return True
        return False

    @staticmethod
    def get_common_responses(responses):
        # returns the common responses out of repeated responses
        return np.sign(np.sum(responses, axis=0))

    @staticmethod
    def get_measured_rels(responses):
        # returns array of measured reliabilities of instance
        return np.abs(np.sum(responses, axis=0))
