"""This module provides a learner exploiting different reliabilities of challenges evaluated several times on an
XOR Arbiter PUF. It is based on the work from G. T. Becker in "The Gap Between Promise and Reality: On the Insecurity
of XOR Arbiter PUFs". The learning algorithm applies Covariance Matrix Adaptation Evolution Strategies from
N. Hansen in "The CMA Evolution Strategy: A Comparing Review".
"""
import sys
import contextlib
import numpy as np
from scipy.special import gamma
from scipy.linalg import norm
import cma

from pypuf import tools
from pypuf.learner.base import Learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


class ReliabilityBasedCMAES(Learner):
    """This class implements a learner for XOR LTF arrays. Thus, by means of a CMAES algorithm a model
    is created similar to the original LTF array. This process is based on the information of reliability
    originating from multiple repeatedly evaluated challenges.
    """

    # Constants
    CONST_EPSILON = 0.1
    FREQ_ABORTION_CHECK = 50
    FREQ_LOGGING = 1
    APPROX_CHALLENGE_NUM = 10000
    THRESHOLD_DIST = 0.25

    def __init__(self, training_set, k, n, transform, combiner,
                 pop_size, limit_stag, limit_iter, random_seed, logger):
        """Initialize a Reliability based CMAES Learner for the specified LTF array

        :param training_set:    Training set, a data structure containing repeated challenge response pairs
        :param k:               Width, the number of parallel LTFs in the LTF array
        :param n:               Length, the number stages within the LTF array
        :param transform:       Transformation function, the function that modifies the input within the LTF array
        :param combiner:        Combiner, the function that combines particular chains' outputs within the LTF array
        :param pop_size:        Population size, the number of sampled points of every CMAES iteration
        :param limit_stag:      Stagnation limit, the maximal number of stagnating iterations within the CMAES
        :param limit_iter:      Iteration limit, the maximal number of iterations within the CMAES
        :param random_seed:     PRNG seed used by the CMAES algorithm for sampling solution points
        :param logger:          Logger, the instance that logs detailed information every learning iteration
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

        if 2**n < self.APPROX_CHALLENGE_NUM:
            self.APPROX_CHALLENGE_NUM = 2 ** n

    def learn(self):
        """Compute a model according to the given LTF Array parameters and training set
        Note that this function can take long to return
        :return: pypuf.simulation.arbiter_based.LTFArray
                 The computed model.
        """

        def log_state(ellipsoid):
            """Log a snapshot of learning variables while running"""
            if self.logger is None:
                return
            self.logger.debug(
                '%i\t%f\t%f\t%i\t%i\t%s',
                self.num_iterations,
                ellipsoid.sigma,
                fitness(ellipsoid.mean),
                self.num_learned,
                self.num_abortions,
                ','.join(map(str, list(ellipsoid.mean))),
            )

        # Preparation
        epsilon = np.sqrt(self.n) * self.CONST_EPSILON
        fitness = self.create_fitness_function(
            challenges=self.training_set.challenges,
            measured_rels=self.measure_rels(self.training_set.responses),
            epsilon=epsilon,
            transform=self.transform,
            combiner=self.combiner,
        )
        normalize = np.sqrt(2) * gamma(self.n / 2) / gamma((self.n - 1) / 2)
        mean_start = np.zeros(self.n)
        step_size_start = 1
        options = {
            'seed': 0,
            'pop': self.pop_size,
            'maxiter': self.limit_i,
            'tolstagnation': self.limit_s,
        }

        # Learn all individual LTF arrays (chains)
        with self.avoid_printing():
            while self.num_learned < self.k:
                aborted = False
                options['seed'] = self.prng.randint(2 ** 32)
                is_same_solution = self.create_abortion_function(
                    chains_learned=self.chains_learned,
                    num_learned=self.num_learned,
                    transform=self.transform,
                    combiner=self.combiner,
                    threshold=self.THRESHOLD_DIST,
                )
                search = cma.CMAEvolutionStrategy(x0=mean_start, sigma0=step_size_start, inopts=options)
                counter = 1
                # Learn individual LTF array using abortion if evolutionary search approximates previous a solution
                while not search.stop():
                    curr_points = search.ask()  # Sample new solution points
                    search.tell(curr_points, [fitness(point) for point in curr_points])
                    self.num_iterations += 1
                    if counter % self.FREQ_LOGGING == 0:
                        log_state(search)
                    counter += 1
                    if counter % self.FREQ_ABORTION_CHECK == 0:
                        if is_same_solution(search.mean):
                            self.num_abortions += 1
                            aborted = True
                            break
                solution = search.result.xbest

                # Include normalized solution, if it is different from previous ones
                if not aborted:
                    self.chains_learned[self.num_learned] = normalize * solution / norm(solution)
                    self.num_learned += 1
                    if self.stops != '':
                        self.stops += ','
                    self.stops += '_'.join(list(search.stop()))

        # Polarize the learned combined LTF array
        majority_responses = self.majority_responses(self.training_set.responses)
        self.chains_learned = self.polarize_chains(
            chains_learned=self.chains_learned,
            challenges=self.training_set.challenges,
            majority_responses=majority_responses,
            transform=self.transform,
            combiner=self.combiner,
        )
        return LTFArray(self.chains_learned, self.transform, self.combiner)

    @staticmethod
    @contextlib.contextmanager
    def avoid_printing():
        """Avoid printing on sys.stdout while learning"""
        save_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')
        yield
        sys.stdout.close()
        sys.stdout = save_stdout

    @staticmethod
    def create_fitness_function(challenges, measured_rels, epsilon, transform, combiner):
        """Return a fitness function on a fixed set of challenges and corresponding reliabilities"""
        this = __class__

        def fitness(individual):
            """Return individuals sorted by their correlation coefficient as fitness"""
            ltf_array = LTFArray(individual[np.newaxis, :], transform, combiner)
            delay_diffs = ltf_array.val(challenges)
            reliabilities = np.zeros(np.shape(delay_diffs))
            indices_of_reliable = np.abs(delay_diffs[:]) > epsilon
            reliabilities[indices_of_reliable] = 1
            correlation = this.calc_corr(reliabilities, measured_rels)
            obj_vals = 1 - (1 + correlation)/2
            return obj_vals

        return fitness

    @staticmethod
    def calc_corr(reliabilities, measured_rels):
        """Return pearson correlation coefficient between reliability arrays of individual and instance"""
        if np.var(reliabilities[:]) == 0:  # Avoid dividing by zero
            return -1
        else:
            return np.corrcoef(reliabilities[:], measured_rels)[0, 1]

    @staticmethod
    def create_abortion_function(chains_learned, num_learned, transform, combiner, threshold):
        """Return an abortion function on a fixed set of challenges and LTFs"""
        this = __class__
        weight_arrays = chains_learned[:num_learned, :]
        learned_ltf_arrays = list(this.build_individual_ltf_arrays(weight_arrays, transform, combiner))

        def is_same_solution(solution):
            """Return True, if the current solution mean within CMAES is similar to a previously learned LTF array"""
            if num_learned == 0:
                return False
            new_ltf_array = LTFArray(solution[np.newaxis, :], transform, combiner)
            for current_ltf_array in learned_ltf_arrays:
                dist = tools.approx_dist(current_ltf_array, new_ltf_array, this.APPROX_CHALLENGE_NUM)
                if dist < threshold or dist > (1 - threshold):
                    return True
            return False

        return is_same_solution

    @staticmethod
    def polarize_chains(chains_learned, challenges, majority_responses, transform, combiner):
        """Return the correctly polarized combined LTF array"""
        model = LTFArray(chains_learned, transform, combiner)
        responses_model = model.eval(challenges)
        num, _ = np.shape(challenges)
        accuracy = np.count_nonzero(responses_model == majority_responses) / num
        polarized_chains = chains_learned
        if accuracy < 0.5:
            polarized_chains[0, :] *= -1
        return polarized_chains

    @staticmethod
    def build_individual_ltf_arrays(weight_arrays, transform, combiner):
        """Return iterator over LTF arrays created out of every individual"""
        pop_size, _ = np.shape(weight_arrays)
        for i in range(pop_size):
            yield LTFArray(weight_arrays[i, np.newaxis, :], transform, combiner)

    @staticmethod
    def majority_responses(responses):
        """Return the common responses out of repeated responses"""
        return np.sign(np.sum(responses, axis=0))

    @staticmethod
    def measure_rels(responses):
        """Return array of measured reliabilities of instance"""
        measured_rels = np.abs(np.sum(responses, axis=0))
        if np.var(measured_rels) == 0:
            raise Exception('The challenges\' reliabilities evaluated on the instance to learn are to high!')
        return measured_rels
