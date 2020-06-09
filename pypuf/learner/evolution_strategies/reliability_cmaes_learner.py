""" This module provides a learner exploiting different reliabilities of challenges
    evaluated several times on an XOR Arbiter PUF. It is based on the work from G. T.
    Becker in "The Gap Between Promise and Reality: On the Insecurity of XOR Arbiter
    PUFs". The learning algorithm applies Covariance Matrix Adaptation Evolution
    Strategies from N. Hansen in "The CMA Evolution Strategy: A Comparing Review".
"""
from cma import CMA
from numpy import array, zeros, mean, float64, average, absolute
from tensorflow import greater, transpose, double, cast, reduce_mean, tensordot, sqrt, reduce_sum, matmul,\
    abs as abs_tf, Variable
from tensorflow_core.python.framework.random_seed import set_seed
from numpy.random.mtrand import RandomState
from scipy.stats import pearsonr, mode
from pypuf.tools import approx_dist
from pypuf.learner.base import Learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


# ==================== Reliability for PUF and MODEL ==================== #

def reliabilities_PUF(response_bits):
    """
        Computes 'Reliabilities' between 0 and 1
        :param response_bits: Array with shape [num_challenges, num_measurements]
    """
    return absolute(average(response_bits, axis=1))


def reliabilities_MODEL(delay_diffs):
    """
        Computes 'Hypothetical Reliabilities'
        :param delay_diffs: Array with shape [num_challenges]
    """
    res = transpose(abs_tf(delay_diffs))
    return cast(res, double)


def tf_pearsonr(x, y):
    """
        Calculates Pearson Correlation Coefficient.
        x and y are matrices with data vectors as columns.
        Return array where index i,j is the pearson correlation of i'th
        column vector in x and the j'th column vector in y.
    """
    centered_x = x - reduce_mean(x, axis=0)
    centered_y = y - reduce_mean(y, axis=0)
    cov_xy = tensordot(transpose(centered_x), centered_y, axes=1)
    auto_cov = sqrt(tensordot(reduce_sum(centered_x**2, axis=0), reduce_sum(centered_y**2, axis=0), axes=0))
    corr = cov_xy / auto_cov
    return corr


# ============================ Learner class ============================ #

class ReliabilityBasedCMAES(Learner):
    """
        This class implements the CMAES algorithm to learn a model of a XOR-Arbiter PUF.
        This process uses information about the (un-)reliability of repeated challenges.

        If a response bit is unstable for a given challenge, it is likely that the delay
        difference is is close to zero: delta_diff < CONST_EPSILON
    """

    def __init__(self, training_set, k, n, transform, combiner, abort_delta, random_seed, logger, max_tries, gpu_id):
        """Initialize a Reliability based CMAES Learner for the specified LTF array

        :param training_set:    Training set, a data structure containing repeated
                                challenge response pairs.
        :param k:               Width, the number of parallel LTFs in the LTF array
        :param n:               Length, the number stages within the LTF array.
        :param transform:       Transformation function, the function that modifies the
                                input within the LTF array.
        :param combiner:        Combiner, the function that combines particular chains'
                                outputs within the LTF array.
        :param abort_delta:     Stagnation value, the maximal delta within *abort_iter*
                                iterations before early stopped.
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
        self.abort_delta = abort_delta
        self.prng = RandomState(random_seed)
        self.chains_learned = zeros((self.k, self.n))
        self.num_tries = 0
        self.stops = ''
        self.logger = logger
        self.max_tries = max_tries
        self.gpu_id = gpu_id
        self.current_fitness = []
        self.fitness_histories = []
        self.pool = None
        self.current_challenges = None

        # Compute PUF Reliabilities. These remain static throughout the optimization.
        self.puf_reliabilities = reliabilities_PUF(self.training_set.responses)

        # Linearize challenges for faster LTF computation (shape=(N,k,n))
        self.linearized_challenges = self.transform(self.training_set.challenges, k=self.k)

    def print_accs(self, es):
        w = es.best.x[:-1]
        a = [
            1 - approx_dist(
                LTFArray(v[:self.n].reshape(1, self.n), self.transform, self.combiner),
                LTFArray(w[:self.n].reshape(1, self.n), self.transform, self.combiner),
                10000,
                self.prng,
            )
            for v in self.training_set.instance.weight_array
        ]
        print(array(a), self.objective(es.best.x))

    def objective(self, state):
        """
            Objective to be minimized. Therefore we use the 'Pearson Correlation
            Coefficient' of the model reliabilities and puf reliabilities.
        """
        # Weights and epsilon have the first dim as number of population
        weights = state[:, :self.n]
        delay_diffs = matmul(weights, self.current_challenges.T)
        model_reliabilities = reliabilities_MODEL(delay_diffs)

        # Calculate pearson coefficient
        x = Variable(model_reliabilities, double)
        y = Variable(self.puf_reliabilities, double)
        corr = tf_pearsonr(x, y)

        # MOD: Calculate correlation with already learned chains
        corr2 = 0
        # Remove punishment for approaching already learned chains
        # if len(self.pool) > 0:
        #     corr2 = abs_tf(tf_pearsonr(array(self.pool).T, transpose(weights)))
        #     mask = greater(corr2, 0.25)
        #     corr2 = reduce_sum(cast(mask, double), axis=0)

        return abs(1 - corr) + corr2

    def test_model(self, model):
        """
            Perform a test using the training set and return the accuracy.
            This function is used at the end of the training phase to determine,
            whether the chains need to be flipped.
        """
        # Since responses can be noisy, we perform majority vote on response bits
        y_true = mode(self.training_set.responses, axis=1)[0].T
        y_test = model.eval(self.training_set.challenges)
        return mean(y_true == y_test)

    def logging_function(self, cma, logger):
        if cma.generation % 10 == 0:
            fitness = cma.best_fitness()
            self.current_fitness.append(fitness)
            logger.info(f'Generation {cma.generation} - fitness {fitness}')

        if cma.termination_criterion_met or cma.generation == 1000:
            sol = cma.best_solution()
            fitness = cma.best_fitness()
            logger.info(f'Final solution at gen {cma.generation}: {sol} (fitness: {fitness})')
            logger.info(f'Termination: {cma.should_terminate(return_details=True)[1]}')

    def learn(self):
        """
            Start learning and return optimized LTFArray and count of failed learning
            attempts.
        """
        # pool: collection of learned chains, meta_data: information about learning
        meta_data, self.pool = {}, []
        meta_data['discard_count'] = {i: [] for i in range(self.k)}
        meta_data['iteration_count'] = {i: [] for i in range(self.k)}
        # For k chains, learn a model and add to pool if "it is new"
        n_chain = 0
        while n_chain < self.k:
            self.current_fitness = []
            print("Attempting to learn chain", n_chain)
            self.current_challenges = array(self.linearized_challenges[:, n_chain, :], dtype=float64)

            set_seed(self.prng.randint(low=0, high=2 ** 32 - 1))
            init_state = list(self.prng.normal(0, 1, size=self.n)) + [2]
            init_state = array(init_state)   # weights = normal_dist; epsilon = 2
            cma = CMA(
                initial_solution=init_state,
                initial_step_size=1.0,
                fitness_function=self.objective,
                termination_no_effect=self.abort_delta,
                callback_function=self.logging_function,
            )

            # Learn the chain (on the GPU)
            # with tf.device('/GPU:%d' % self.gpu_id):
            w, score = cma.search(max_generations=1000)

            # Update meta data about how many iterations it took to find a solution
            meta_data['iteration_count'][n_chain].append(cma.generation)

            w = w[:-1]
            # Flip chain for comparison; invariant of reliability
            w = -w if w[0] < 0 else w

            # Check if learned model (w) is a 'new' chain (not correlated to other chains)
            for i, v in enumerate(self.pool):
                if abs_tf(pearsonr(w, v)[0]) > 0.5 and self.num_tries < self.max_tries:
                    meta_data['discard_count'][n_chain].append(i)
                    self.num_tries += 1
                    break
            else:
                self.pool.append(w)
                self.fitness_histories.append(self.current_fitness)
                n_chain += 1
                self.num_tries = 0

        # Test LTFArray. If accuracy < 0.5, we flip the first chain, hence the output bits
        model = LTFArray(array(self.pool), self.transform, self.combiner)
        if self.test_model(model) < 0.5:
            self.pool[0] = - self.pool[0]
            model = LTFArray(array(self.pool), self.transform, self.combiner)
        meta_data['fitness_histories'] = self.fitness_histories

        return model, meta_data
