import numpy as np
from pypuf.simulation.arbiter_based.ltfarray import LTFArray

class reliability_based_CMA_ES():

    # recommended properties of parameters:
    #   pop_size=30
    #   parent_size=10 (>= 0.3*pop_size)
    #   priorities: linear low decreasing
    def __init__(self, instance, pop_size, parent_size, priorities,
                 mu_weight=0, sigma_weight=1):
        self.instance = instance            # simulation instance to be modelled
        self.mu_weight = mu_weight          # mean value of weights
        self.sigma_weight = sigma_weight    # sd of weights
        self.individuals = np.zeros((instance.n, parent_size))
        # mean, step_size,  pop_size,   parent_size,    priorities, cov_matrix,   path_cm,    path_ss
        # m,    sigma,      lambda,     mu,             w_i,        C,            p_c,        p_sigma
        self.mean = np.zeros(instance.n) # mean vector of distribution
        self.step_size = sigma_weight / 2   # distance to next distribution
        self.pop_size = pop_size        # number of individuals per generation
        self.parent_size = parent_size  # number of considered individuals
        self.priorities = priorities    # array of consideration proportion
        self.cov_matrix = np.identity(instance.n)  # shape of distribution
        self.path_cm = 0    # cumulated evolution path of covariance matrix
        self.path_ss = 0    # cumulated evolution path of step size
        # auxiliary constants
        self.mu_w = 1 / np.sum(np.square(priorities))
        self.c_mu = self.mu_w / instance.n**2
        self.d_sigma = 1 + np.sqrt(self.mu_w / instance.n)
        self.c_1 = 2 / instance.n**2
        self.c_c = 4 / instance.n
        self.c_sigma = 4 / instance.n
        assert len(priorities)==parent_size and np.sum(priorities)==1
        assert (self.c_1 + self.mu_w <= 1)

    def learn(self):
        terminate = False
        while not terminate:
            self.individuals = self.reproduce(self.mean, self.cov_matrix,
                        self.pop_size, self.step_size)
            self.mean = self.update_mean()
            self.path_cm = self.cumulation_for_cm()
            self.path_ss = self.cumulation_for_ss()
            self.cov_matrix = self.update_cm()
            self.step_size = self.update_ss()


    # primary methods
    @staticmethod
    def reproduce(mean, cov_matrix, pop_size, step_size):
        # returns a new generation of individuals as 2D array (pop_size, n)
        mutations = np.random.multivariate_normal(np.zeros(np.shape(mean)),
                                                  cov_matrix, pop_size).T
        duplicated_mean = np.tile(np.matrix(mean).T, (1, pop_size))
        return duplicated_mean + (step_size * mutations)

    @staticmethod
    def update_mean(old_mean, step_size, parent):
        # returns mean of a new population as array (n)
        return old_mean + step_size*parent

    @staticmethod
    def cumulation_for_cm(path_cm, c_c, path_ss, n, mu_w, parent):
        # returns cumulated evolution path of covariance matrix
        path_cm *= (1-c_c)
        if(np.linalg.norm(path_ss) < 1.5 * np.sqrt(n)):
            path_cm += np.sqrt(1 - (1-c_c)**2) * np.sqrt(mu_w) * parent
        return path_cm

    @staticmethod
    def cumulation_for_ss(path_ss, c_sigma, mu_w, parent):
        # returns cumulated evolution path of step-size
        path_ss = (1-c_sigma) * path_ss + np.sqrt(1 - (1-c_sigma)**2)\
                                          * np.sqrt(mu_w) * parent
        return path_ss

    @staticmethod
    def update_cm(cov_matrix, c_1, c_mu, path_cm, w_i, y_i):
        # returns covariance matrix of a new population (pop_size, pop_size)
        pass

    @staticmethod
    def update_ss(step_size, c_sigma, d_sigma, path_ss):
        # returns step-size of a new population
        pass


    # secondary methods
    @staticmethod
    def get_parent(sorted_individuals, parent_size, priorities):
        # returns the weighted sum of the fittest individuals
        parent = np.empty(np.shape(sorted_individuals)[1])
        for i in range(parent_size):
            parent += priorities[i] * sorted_individuals[i,:]
        return parent

    @staticmethod
    def fitness(instance, challenges, reliabilities, individuals):
        """
        build ltfarrays out of individuals
        get delay differences from ltfarrays for challenges
        get reliability vector out of delay differences
        get correlation coefficient of reliability vectors
        sort individuals by correlation coefficients
        return sorted individuals
        """
        # returns individuals sorted by their fitness
        pass


    def build_ltf_arrays(individuals):
        # returns iterator over ltf_arrays created out of every individual
        transform = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        pop_size = np.shape(individuals)[1]
        for i in range(pop_size):
            yield LTFArray(individuals.T[:, i], transform, combiner,
                           bias=False)

    def get_delay_differences(ltf_arrays, pop_size, challenges, challenge_num):
        # returns 2D array of delay differences for all challenges on every
        #   individual
        delay_diffs = np.empty(pop_size, challenge_num)
        for i, ltf_array in ltf_arrays:
            for j, challenge in challenges:
                delay_diffs[i][j] = ltf_array.val(challenge)
        return delay_diffs

    def get_reliabilities(delay_diffs, epsilon):
        # returns 2D array of reliabilities for all challenges on every individual
        reliabilities = np.zeros(np.shape(delay_diffs))
        indices_of_unreliable = delay_diffs > -epsilon and delay_diffs < epsilon
        reliabilities[indices_of_unreliable] = 1
        return reliabilities

    def get_correlations(reliabilities, measured_rels):
        # returns array of pearson correlation coefficients between reliability
        #   array of individual and instance for all individuals
        pop_size = np.shape(reliabilities[0])
        correlations = np.empty(pop_size)
        for i in range(pop_size):
            correlations[i] = np.corrcoef(reliabilities[i,:], measured_rels)[0, 1]
        return correlations

    def sort_individuals(individuals, correlations):
        # returns 2D array of individuals as given from input, but sorted through
        #   correlation coefficients
        pass
