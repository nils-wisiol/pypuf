""" This module provides a learner exploiting different reliabilities of challenges
    evaluated several times on an XOR Arbiter PUF. It is based on the work from G. T.
    Becker in "The Gap Between Promise and Reality: On the Insecurity of XOR Arbiter
    PUFs". The learning algorithm applies Covariance Matrix Adaptation Evolution
    Strategies from N. Hansen in "The CMA Evolution Strategy: A Comparing Review".
"""
import cma
import numpy as np

from scipy.stats import pearsonr

from pypuf.learner.base import Learner
from pypuf.tools import transform_challenge_11_to_01
from pypuf.simulation.arbiter_based.ltfarray import LTFArray

# ==================== Reliability for PUF and MODEL ==================== #

def reliabilities_PUF(response_bits):
    """
        Computes 'Reliabilities' according to [Becker].
        :param response_bits: Array with shape [num_challenges, num_measurements]
    """
    # Convert to 0/1 from 1/-1
    response_bits = np.array(response_bits, dtype=np.int8)
    if (-1 in response_bits):
        response_bits = transform_challenge_11_to_01(response_bits)
    return np.abs(response_bits.shape[1]/2 - np.sum(response_bits, axis=1))

def reliabilities_MODEL(delay_diffs, EPSILON=3):
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

        #TODO linearize challenges to use them in id transform for LTF model

        self.puf_reliabilities = reliabilities_PUF(self.training_set.responses)

    def objective(self, weights):
        """
            Objective to be minimized. Therefore we use the 'Pearson Correlation
            Coefficient' of the model reliabilities and puf reliabilities.
        """
        model = LTFArray(weights[np.newaxis, :], self.transform, self.combiner)
        delay_diffs = model.val(self.training_set.challenges)
        model_reliabilities = reliabilities_MODEL(delay_diffs)
        corr = pearsonr(model_reliabilities, self.puf_reliabilities)
        return np.abs(1 - corr[0])


    def learn(self):
        options = {
            'seed': self.prng.randint(2 ** 32),
            #'timeout': "2.5 * 60**2"
            'maxiter': self.limit_i,
            #'tolstagnation': self.limit_s,
        }
        es = cma.CMAEvolutionStrategy(np.zeros(self.n), 1, inopts=options)
        es.optimize(self.objective)
        w = es.best.x
        model = LTFArray(w[np.newaxis, :], self.transform, self.combiner)
        return model

