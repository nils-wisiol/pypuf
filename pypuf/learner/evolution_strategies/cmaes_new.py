""" This module provides a learner exploiting different reliabilities of challenges
    evaluated several times on an XOR Arbiter PUF. It is based on the work from G. T.
    Becker in "The Gap Between Promise and Reality: On the Insecurity of XOR Arbiter
    PUFs". The learning algorithm applies Covariance Matrix Adaptation Evolution
    Strategies from N. Hansen in "The CMA Evolution Strategy: A Comparing Review".
"""
import cma
import numpy as np

from tools import transform_challenge_11_to_01 as 11_to_01


# ==================== Reliability for PUF and MODEL ==================== #

def reliabilities_PUF(response_bits):
    """
        Computes 'Reliabilities' according to [Becker].
        :param response_bits: Array with shape [num_challenges, num_measurements]
    """
    # Convert to 0/1 from 1/-1
    if (-1 in response_bits):
        response_bits = 11_to_01(response_bits)
    return np.abs(response_bits.shape[1]/2 - np.sum(response_bits, axis=0))

def reliabilities_MODEL(delay_diffs, EPSILON=0.1):
    """
        Computes 'Hypothical Reliabilities' according to [Becker].
        :param delay_diffs: Array with shape [num_challenges]
    """
    return np.abs(delay_diffs) > EPSILON

# ============================ Learner class ============================ #

class ReliabilityBasedCMAES(Learner):
    """
        This class implements the CMAES algorithm to learn a model of a XOR-Arbiter PUF.
        This process uses information about the (un-)reliability of repeated challenges.

        If a response bit is unstable for a given challenge, it is likely that the delay
        difference is is close to zero: delta_diff < CONST_EPSILON
    """
    # Constants
    CONST_FREQ_ABORTION_CHECK = 50
    CONST_FREQ_LOGGING = 1
    CONST_THRESHOLD_DIST = 0.25

    def __init__(self, training_set, k, n, transform, combiner,
                 pop_size, limit_stag, limit_iter, random_seed, logger):
        """Initialize a Reliability based CMAES Learner for the specified LTF array

        :param training_set:    Training set, a data structure containing repeated
                                challenge response pairs
        :param k:               Width, the number of parallel LTFs in the LTF array
        :param n:               Length, the number stages within the LTF array
        :param transform:       Transformation function, the function that modifies the
                                input within the LTF array
        :param combiner:        Combiner, the function that combines particular chains'
                                outputs within the LTF array
        :param pop_size:        Population size, the number of sampled points of every
                                CMAES iteration
        :param limit_stag:      Stagnation limit, the maximal number of stagnating
                                iterations within the CMAES
        :param limit_iter:      Iteration limit, the maximal number of iterations within
                                the CMAES
        :param random_seed:     PRNG seed used by the CMAES algorithm for sampling
                                solution points
        :param logger:          Logger, the instance that logs detailed information every
                                learning iteration
        """
        self.training_set = training_set
        self.k = k
        self.n = n
        self.transform = transform
        self.combiner = combiner
        self.pop_size = pop_size
        self.limit_s = limit_stag
        self.limit_i = limit_iter
        self.prng = np.random.RandomState(random_seed)
        self.chains_learned = np.zeros((self.k, self.n))
        self.num_iterations = 0
        self.stops = ''
        self.num_abortions = 0
        self.num_learned = 0
        self.logger = logger

        self.puf_reliabilities = reliabilities_PUF(self.training_set.responses)

    def fitness(self, weights):
        """
            Fitness of a model given by weights. Therefore we use the 'Pearson Correlation
            Coefficient' of the model reliabilities and puf reliabilities.
        """
        model = LTFArray(weights[np.newaxis, :], self.transform, self.combiner)
        delay_diffs = model.val(self.training_set.challenges)
        model_reliabilities = reliabilities_MODEL(delay_diffs)
        return np.corrcoef(model_reliabilities, self.puf_reliabilities)


    def learn(self):
        options = {
            'seed': self.prng.randint(2 ** 32),
            'pop': self.pop_size,
            'maxiter': self.limit_i,
            'tolstagnation': self.limit_s,
        }
        es = cma.CMAEvolutionStrategy(np.zeros(self.n), 0.5, inopts=options)
        es.optimize(self.fitness)

