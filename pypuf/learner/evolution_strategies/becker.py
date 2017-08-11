import numpy as np
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.evolution_strategies.cma_es import CMA_ES

class Reliability_based_CMA_ES():

    # recommended properties of parameters:
    #   pop_size=30
    #   parent_size=10 (>= 0.3*pop_size)
    #   priorities: linearly low decreasing
    def __init__(self, k, n, transform, combiner, challenges, responses, repeat, precision, pop_size, parent_size, priorities,
                 prng=np.random.RandomState()):
        self.k = k                                              # number of XORed LTFs
        self.n = n                                              # length of LTFs plus 1 because of epsilon
        self.transform = transform                              # function for modifying challenges
        self.combiner = combiner                                # function for combining the particular LTFArrays
        self.challenges = challenges                            # set of challenges applied on instance to learn
        self.responses = responses                              # evaluated responses of the challenges on instance
        self.repeat = repeat                                    # frequency of same repeated challenge
        self.different_LTFs = np.zeros((self.k, self.n-1))      # all currently learned LTFs (without epsilon)
        self.num_of_LTFs = 0                                    # number of different learned LTFs
        self.measured_rels = self.get_measured_rels(self.responses) # measured reliabilities (instance)
        # parameters for CMA_ES
        self.prng = prng                                        # pseudo random number generator (CMA_ES)
        self.precision = precision                              # intended scale of step-size to achieve (CMA_ES)
        self.pop_size = pop_size                                # number of models sampled each iteration (CMA_ES)
        self.parent_size = parent_size                          # number of considered models each iteration (CMA_ES)
        self.priorities = priorities                            # array of consideration proportion (CMA_ES)

    def learn(self):
        # this is the general learning method
        # returns an XOR-LTFArray with nearly the same behavior as learned instance
        fitness_function = self.get_fitness_function(self.challenges, self.measured_rels, self.transform, self.combiner)
        while self.num_of_LTFs < self.k:
            cma_es = CMA_ES(fitness_function, self.precision, self.n, self.pop_size, self.parent_size,
                            self.priorities, self.prng)
            new_LTF = cma_es.evolutionary_search()[:-1]     # remove epsilon
            if self.is_different_LTF(new_LTF, self.different_LTFs, self.num_of_LTFs, self.challenges, self.transform,
                                     self.combiner):
                self.different_LTFs[self.num_of_LTFs] = new_LTF
                self.num_of_LTFs += 1
        common_responses = self.get_common_responses(self.responses)
        self.different_LTFs = self.set_pole_of_LTFs(self.different_LTFs, self.challenges, common_responses,
                                                    self.transform, self.combiner)
        return LTFArray(self.different_LTFs, LTFArray.transform_atf, LTFArray.combiner_xor, bias=False)

    @staticmethod
    def get_fitness_function(challenges, measured_rels, transform, combiner):
        # returns a fitness function on a fixed set of challenges and corresponding reliabilities
        becker = __class__

        def fitness(individuals):
            # returns individuals sorted by their correlation coefficient as fitness
            pop_size = np.shape(individuals)[0]
            built_LTFArrays = becker.build_LTFArrays(individuals[:, :-1], transform, combiner)
            delay_diffs = becker.get_delay_differences(built_LTFArrays, pop_size, challenges)
            epsilons = individuals[:, -1]
            reliabilities = becker.get_reliabilities(delay_diffs, epsilons)
            correlations = becker.get_correlations(reliabilities, measured_rels)
            return correlations

        return fitness

    @staticmethod
    def is_different_LTF(new_LTF, different_LTFs, num_of_LTFs, challenges, transform, combiner):
        # returns True, iff new_LTF is different from previously learned LTFs
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
        delay_diffs = np.zeros((pop_size, np.shape(challenges)[0]))
        for i, built_LTFArray in enumerate(built_LTFArrays):
            delay_diffs[i, :] = built_LTFArray.val(challenges)
        return delay_diffs

    @staticmethod
    def get_reliabilities(delay_diffs, epsilons):
        # returns 2D array of reliabilities for all challenges on every individual
        reliabilities = np.zeros(np.shape(delay_diffs))
        for i in range(np.shape(epsilons)[0]):
            indices_of_reliable = np.abs(delay_diffs[i, :]) > np.abs(epsilons[i])
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
        # returns True, iff 2 response arrays are more than 75% equal
        num_of_LTFs, challenge_num = np.shape(responses_diff_LTFs)
        for i in range(num_of_LTFs):
            differences = np.sum(np.abs(responses_new_LTF[:] - responses_diff_LTFs[i, :])) / 2
            if differences < 0.25*challenge_num or differences > 0.75*challenge_num:
                return True
        return False

    @staticmethod
    def get_common_responses(responses):
        return np.sign(np.sum(responses, axis=0))

    @staticmethod
    def get_measured_rels(responses):
        # returns array of measured reliabilities of instance
        return np.abs(np.sum(responses, axis=0))
