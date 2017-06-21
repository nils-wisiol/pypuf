import numpy as np
from pypuf import simulation

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
                        self.parent_size, self.pop_size, self.step_size)
            self.mean = self.update_mean()
            self.path_cm = self.cumulation_for_cm()
            self.path_ss = self.cumulation_for_ss()
            self.cov_matrix = self.update_cm()
            self.step_size = self.update_ss()


    # primary methods
    @staticmethod
    def reproduce(mean, cov_matrix, parent_size, pop_size, step_size):
        mutations = np.random.multivariate_normal(mean,
                                                  cov_matrix, parent_size).T
        mean_as_matrix = np.tile(np.matrix(mean).T, (pop_size, 1))
        return mean_as_matrix + step_size * mutations

    @staticmethod
    def update_mean():
        pass

    @staticmethod
    def cumulation_for_cm():
        pass

    @staticmethod
    def cumulation_for_ss():
        pass

    @staticmethod
    def update_cm():
        pass

    @staticmethod
    def update_ss():
        pass


    # secondary methods
    def fitness(self):
        pass

    def reliability(self, l):
        #l/2 - r[i][j]
        pass
