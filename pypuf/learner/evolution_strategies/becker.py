import numpy as np
import itertools
from pypuf import tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
import pypuf.learner.evolution_strategies.CMA_ES as CMA_ES

class Reliability_based_CMA_ES():

    # recommended properties of parameters:
    #   pop_size=30
    #   parent_size=10 (>= 0.3*pop_size)
    #   priorities: linearly low decreasing
    def __init__(self, k, n, challenges, responses, repeat, precision, pop_size, parent_size, priorities, random_seed):
        self.k = k                                              # number of XORed LTFs
        self.n = n                                              # length of LTFs plus 1 because of epsilon
        self.challenges = challenges                            # set of challenges applied on instance to learn
        self.responses = responses                              # evaluated responses of the challenges on instance
        self.repeat = repeat                                    # frequency of same repeated challenge
        self.different_LTFs = np.zeros((self.k, self.n))        # all currently learned LTFs
        self.num_of_LTFs = 0                                    # number of different learned LTFs
        self.challenge_num = np.shape(self.challenges)[0]       # number of challenges used
        self.measured_rels = self.get_measured_rels(self.responses, self.repeat)    # measured reliabilities (instance)
        # parameters for CMA_ES
        self.prng = np.random.RandomState(seed=random_seed)     # pseudo random number generator (CMA_ES)
        self.precision = precision                              # intended scale of step-size to achieve (CMA_ES)
        self.pop_size = pop_size                                # number of models sampled each iteration (CMA_ES)
        self.parent_size = parent_size                          # number of considered models each iteration (CMA_ES)
        self.priorities = priorities                            # array of consideration proportion (CMA_ES)

    def learn(self):
        # this is the main learning method
        # returns XOR-LTFArray with nearly the same behavior as learned instance
        # correlations = self.fitness(self.challenges, self.challenge_num, self.reliabilities, individuals)
        fitness_function = self.get_fitness_function(self.challenges, self.challenge_num, self.measured_rels)
        while self.num_of_LTFs < self.k:
            cma_es = CMA_ES(fitness_function, self.precision, self.n, self.pop_size, self.parent_size,
                            self.priorities, self.prng)
            new_LTF = cma_es.evolutionary_search()
            if self.is_different_LTF(new_LTF):
                self.different_LTFs[self.num_of_LTFs] = new_LTF
                self.num_of_LTFs += 1
        self.different_LTFs = self.set_pole_of_LTFs(self.different_LTFs, self.challenges, self.responses)
        return LTFArray(self.different_LTFs, LTFArray.transform_atf, LTFArray.combiner_xor, bias=False)

    def is_different_LTF(self, new_LTF):
        # returns True iff new_LTF is different from previously learned LTFs
        if self.num_of_LTFs == 0:
            return True
        weight_arrays = self.different_LTFs[:self.num_of_LTFs, :]
        new_LTFArray = LTFArray(new_LTF[:, :-1], LTFArray.transform_atf, LTFArray.combiner_xor)
        different_LTFArrays = self.build_LTFArrays(weight_arrays[:, :-1])
        responses = np.zeros((self.num_of_LTFs, self.challenge_num))
        responses[0, :] = new_LTFArray.eval(self.challenges)
        for i, current_LTF in enumerate(different_LTFArrays):
            responses[i+1, :] = current_LTF.eval(self.challenges)
        return not self.is_correlated(responses)

    @staticmethod
    def get_fitness_function(challenges, challenge_num, measured_rels):
        # returns a fitness function on a fixed set of challenges and corresponding reliabilities
        becker = __class__

        def fitness(individuals):
            # returns individuals sorted by their correlation coefficient as fitness
            pop_size = np.shape(individuals)[0]
            built_LTFArrays = becker.build_LTFArrays(individuals[:, :-1])
            delay_diffs = becker.get_delay_differences(built_LTFArrays, pop_size, challenges, challenge_num)
            epsilons = individuals[:, -1]
            reliabilities = becker.get_reliabilities(delay_diffs, epsilons)
            correlations = becker.get_correlations(reliabilities, measured_rels)
            return correlations

        return fitness


    # methods for calculating fitness
    @staticmethod
    def build_LTFArrays(weight_arrays):
        # returns iterator over ltf_arrays created out of every individual
        pop_size = np.shape(weight_arrays)[0]
        for i in range(pop_size):
            yield LTFArray(weight_arrays[i, np.newaxis, :], LTFArray.transform_atf, LTFArray.combiner_xor, bias=False)

    @staticmethod
    def get_delay_differences(built_LTFArrays, pop_size, challenges, challenge_num):
        # returns 2D array of delay differences for all challenges on every individual
        delay_diffs = np.empty((pop_size, challenge_num))
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
        for i in range(pop_size):
            if np.any(reliabilities[i, :]):
                correlations[i] = np.corrcoef(reliabilities[i, :], measured_rels)[0, 1]
        return correlations


    # helping methods
    @staticmethod
    def set_pole_of_LTFs(different_LTFs, challenges, responses):
        # returns the correctly polarized XOR-LTFArray
        challenge_num = np.shape(challenges)[0]
        xor_LTFArray = LTFArray(different_LTFs, LTFArray.transform_id, LTFArray.combiner_xor)
        responses_model = xor_LTFArray.eval(challenges)
        difference = np.sum(np.abs(responses - responses_model))
        if difference > challenge_num:
            different_LTFs[0, :] *= -1
        return different_LTFs

    @staticmethod
    def is_correlated(responses):
        # returns True iff 2 response arrays are more than 75% equal
        (num_of_LTFs, challenge_num) = np.shape(responses)
        for i in range(1, num_of_LTFs):
            differences = np.sum(np.abs(responses[0, :] - responses[i, :])) / 2
            if differences < 0.25*challenge_num or differences > 0.75*challenge_num:
                return True
        return False

    @staticmethod
    def get_measured_rels(responses, repeat):
        # returns array of measured reliabilities of instance
        return np.abs(np.sum(responses, axis=0)) / repeat
