""" This module provides a learner exploiting different reliabilities of challenges
    evaluated several times on an XOR Arbiter PUF. It is based on the work from G. T.
    Becker in "The Gap Between Promise and Reality: On the Insecurity of XOR Arbiter
    PUFs". The learning algorithm applies Covariance Matrix Adaptation Evolution
    Strategies from N. Hansen in "The CMA Evolution Strategy: A Comparing Review".
"""
import cma
import numpy as np

from scipy.stats import pearsonr

from pypuf.tools import approx_dist, transform_challenge_11_to_01
from pypuf.bipoly import BiPoly
from pypuf.learner.base import Learner
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

    def __init__(self, training_set, k, n, transform, combiner,
                 pop_size, abort_delta, abort_iter, random_seed, logger):
        """Initialize a Reliability based CMAES Learner for the specified LTF array

        :param training_set:    Training set, a data structure containing repeated
                                challenge response pairs.
        :param k:               Width, the number of parallel LTFs in the LTF array
        :param n:               Length, the number stages within the LTF array.
        :param transform:       Transformation function, the function that modifies the
                                input within the LTF array.
        :param combiner:        Combiner, the function that combines particular chains'
                                outputs within the LTF array.
        :param pop_size:        Population size, the number of sampled points of every
                                CMAES iteration.
        :param abort_delta:     Stagnation value, the maximal delta within *abort_iter*
                                iterations before early stopped.
        :param abort_iter:      Stagnation iteration limit, the window size of iterations
                                where *abort_delta* is calculated.
        :param random_seed:     PRNG seed used by the CMAES algorithm for sampling
                                solution points.
        :param logger:          Logger, the instance that logs detailed information every
                                learning iteration.
        """
        self.training_set = training_set
        self.k = k
        self.n = n
        self.transform = transform
        self.combiner = combiner
        self.pop_size = pop_size
        self.abort_delta = abort_delta
        self.abort_iter = abort_iter
        self.prng = np.random.RandomState(random_seed)
        self.chains_learned = np.zeros((self.k, self.n))
        self.num_iterations = 0
        self.stops = ''
        self.num_abortions = 0
        self.num_learned = 0
        self.logger = logger

        # Compute PUF Reliabilities. These remain static throughout the optimization.
        self.puf_reliabilities = reliabilities_PUF(self.training_set.responses)

        # Linearize challenges for faster LTF computation (shape=(N,k,n))
        self.linearized_challenges = self.transform(self.training_set.challenges,
                                                    k=self.k)


    def print_accs(self, es):
        w = es.best.x[:-1]
        #print(es.fit.hist)
        a = [
            1 - approx_dist(
                LTFArray(v[:self.n].reshape(1,self.n), self.transform, self.combiner),
                LTFArray(w[:self.n].reshape(1,self.n) ,self.transform, self.combiner),
                10000,
                np.random.RandomState(12345)
            )
            for v in self.training_set.instance.weight_array
            ]
        print(np.array(a), self.objective(es.best.x))

    def is_iteration_stagnated(self, es):
        """
            Abort criteria. This function is called after each optimization iteration.
            When True is returned, the optimization is stopped and the current state is
            returned.
        """
        fun_history = es.fit.hist[-self.abort_iter:]
        if (len(es.fit.hist) < 100):
            return False
        if (np.abs(max(fun_history) - min(fun_history)) < self.abort_delta):
            return True
        return False

    def objective(self, state):
        """
            Objective to be minimized. Therefore we use the 'Pearson Correlation
            Coefficient' of the model reliabilities and puf reliabilities.
        """
        weights = state[:self.n]
        epsilon = state[-1]
        model = LTFArray(weights[np.newaxis, :], 'id', self.combiner)
        delay_diffs = model.val(self.current_challenges)
        model_reliabilities = reliabilities_MODEL(delay_diffs, EPSILON=epsilon)
        corr = pearsonr(model_reliabilities, self.puf_reliabilities)
        return np.abs(1 - corr[0])


    def learn(self):
        """
            Start learning and return optimized LTFArray.
        """
        pool = []
        # For k chains, learn a model and add to pool if "it is new"
        n_chain = 0
        while n_chain < self.k:
            print("Attempting to learn chain", n_chain)
            self.current_challenges = self.linearized_challenges[:, n_chain, :]

            cma_options = {
                'seed': self.prng.randint(2 ** 32),
                'termination_callback': self.is_iteration_stagnated
            }
            init_state = list(self.prng.normal(0, 1, size=self.n)) + [2]
            init_state = np.array(init_state) # weights = normal_dist; epsilon = 2
            es = cma.CMAEvolutionStrategy(init_state, 1, inopts=cma_options)
            es.optimize(self.objective, callback=self.print_accs)
            w = es.best.x[:self.n]
            # Flip chain for comparison; invariant of reliability
            w_comp = -w if w[0] < 0 else w

            # Check if learned model (w) is a 'new' chain (not correlated to other chains)
            for v in pool:
                if (np.abs(pearsonr(w_comp, v)[0]) > 0.5):
                    break
            else:
                pool.append(w)
                n_chain += 1

        model = LTFArray(np.array(pool), self.transform, self.combiner)
        return model

