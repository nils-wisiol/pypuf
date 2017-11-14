"""
This module provides an experiment class which implements an experiment in order to find an approximately  minimal
number of votes for a Majority Vote Arbiter PUF which satisfy the chosen desired_stability and
overall_desired_stability.
"""
import numpy as np
from numpy.random import RandomState
from pypuf import tools
from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.ltfarray import SimulationMajorityLTFArray, LTFArray, NoisyLTFArray


class ExperimentMajorityVoteFindVotes(Experiment):
    """
    This experiment can be used to find an approximately  minimal number of votes for a Majority Vote Arbiter PUF which
    satisfy the chosen desired_stability and overall_desired_stability.
    """

    def __init__(self, log_name, n, k, challenge_count, seed_instance, seed_instance_noise, transformation,
                 combiner, mu, sigma, sigma_noise_ratio, seed_challenges, desired_stability, overall_desired_stability,
                 minimum_vote_count, iterations, bias=None):
        """
        :param log_name: string
                         The prefix of the self.progress_logger.
        :param n: int
                  The number of stages of the PUF.
        :param k: int
                  The number different LTFArrays of the SimulationMajorityLTFArray.
        :param challenge_count: int
                                The number of challenges which are used to evaluate the PUF.
        :param seed_instance: int
                              The seed which is used to initialize the pseudo-random number generator
                              which is used to generate the stage weights for the arbiter PUF simulation.
        :param seed_instance_noise: int
                                    The random seed which is used to initialize the pseudo-random number
                                    generator which is used to generate the noise for the arbiter PUF simulation.
        :param transformation: A function: array of int with shape(N,k,n), int number of PUFs k -> shape(N,k,n)
                               The function transforms input challenges in order to increase resistance against attacks.
        :param combiner: A function: array of int with shape(N,k,n) -> array of in with shape(N)
                         The functions combines the outputs of k PUFs to one bit results,
                         in oder to increase resistance against attacks.
        :param mu: float
                   Mean (“centre”) of the stage weight distribution of the PUF instance simulation.
        :param sigma: float
                      Standard deviation of the stage weight distribution of the PUF instance simulation.
        :param sigma_noise_ratio: float
                                  The noisiness factor which is used to scale sigma_noise. The value sigma_noise
                                  is the standard deviation of the noise distribution of the PUF instance simulation.
        :param seed_challenges: int
                                The seed which is used to initialize the pseudo-random number generator which
                                is used to generate challenges.
        :param desired_stability: float
                                  The number which is used to decide whether a PUF is stable or not.
        :param overall_desired_stability: float
                                          The relative frequency of challenges which are greater equal than the
                                          desired_stability.
        :param minimum_vote_count: int
                       That is the minimum number of votes which are used to evaluate the SimulationMajorityLTFArray
                       instance.
        :param iterations: int
                           The number of evaluations of the SimulationMajorityLTFArray instance which are used
                           to check the desired_stability.
        :param bias: None, float or a one dimensional list of float with shape (k)
                     This bias value or list of bias values will be appended to the weight_array.
                     Use a single value if you want the same bias for all weight_vectors.
        """
        super().__init__(
            log_name='%s.0x%x_0_%i_%i_%i_%s_%s' % (
                log_name,
                seed_instance,
                n,
                k,
                challenge_count,
                transformation.__name__,
                combiner.__name__,
            ),
        )
        self.n = n
        self.k = k
        self.N = challenge_count
        self.seed_instance = seed_instance
        self.seed_instance_noise = seed_instance_noise
        self.seed_challenges = seed_challenges
        self.transformation = transformation
        self.combiner = combiner
        self.mu = mu
        self.sigma = sigma
        self.sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, 1, sigma_noise_ratio)
        self.bias = bias
        self.desired_stability = desired_stability
        self.overall_desired_stability = overall_desired_stability
        self.minimum_vote_count = minimum_vote_count
        self.maximum_vote_count = 0  # Upper bound for binary search calculated in run()
        self.vote_count = 0  # That is calculated during run()
        self.result_overall_stab = 0.0
        # This saves the first found over all stability which satisfy the overall_desired_stability
        self.first_result_overall_stab = 0.0
        self.result_vote_count = 0
        self.iterations = iterations
        self.overall_stab = 0.0  # That is the overall_stab for vote_count calculated during run()

    def run(self):
        """
        Searches for the minimal number of votes which satisfy self.desired_stability and
        self.overall_desired_stability. The number of votes for a try is given by the interval between self.bottom and
        self.top.
        """
        # Random number generators
        instance_prng = RandomState(self.seed_instance)
        noise_prng = RandomState(self.seed_instance_noise)
        challenge_prng = RandomState(self.seed_challenges)

        # Weight array for the instance which should be learned
        weight_array = LTFArray.normal_weights(self.n, self.k, self.mu, self.sigma, random_instance=instance_prng)

        if isinstance(self.bias, type('list')):
            self.bias = np.reshape(np.array(self.bias), (self.k, 1))

        self.vote_count = 0
        # Search upper bound for binary search
        while self.overall_stab < self.overall_desired_stability:
            self.vote_count = self.vote_count * 2 + 1
            puf_instance = SimulationMajorityLTFArray(weight_array, LTFArray.transform_id,
                                                      LTFArray.combiner_xor, self.sigma_noise,
                                                      random_instance_noise=noise_prng, vote_count=self.vote_count,
                                                      bias=self.bias)
            self.calculate_stabilities(puf_instance, challenge_prng)
        self.maximum_vote_count = self.vote_count
        self.first_result_overall_stab = self.overall_stab
        self.result_vote_count = self.vote_count

        # Binary search loop which is used to find the minimum number of votes in oder to satisfy
        # self.desired_stability and overall_desired_stability
        while self.minimum_vote_count < self.maximum_vote_count:
            # Set the number of vote counts
            self.vote_count = (self.minimum_vote_count + self.maximum_vote_count) // 2
            # self.vote_count must be odd
            if self.vote_count % 2 == 0:
                self.vote_count = self.vote_count + 1

            puf_instance = SimulationMajorityLTFArray(weight_array, LTFArray.transform_id,
                                                      LTFArray.combiner_xor, self.sigma_noise,
                                                      random_instance_noise=noise_prng, vote_count=self.vote_count)
            self.calculate_stabilities(puf_instance, challenge_prng)

            #     overall_stab      vote_count
            msg = '%f\t'            '%i\t' % (self.overall_stab, self.vote_count)

            # Interval adjustment
            if self.overall_stab >= self.overall_desired_stability:
                self.maximum_vote_count = self.vote_count - 1
                self.result_vote_count = self.vote_count
                self.result_overall_stab = self.overall_stab
            else:
                self.minimum_vote_count = self.vote_count + 1

            # If the first result was the best
            if self.result_overall_stab == 0.0:
                self.result_overall_stab = self.first_result_overall_stab

            self.progress_logger.info(msg)

    def analyze(self):
        """
        Summarize the results of the search process.
        """
        #     seed_instance seed_challenges n      k      N      vote_count  overall_stab   measured_time
        msg = '0x%x\t'      '0x%x\t'        '%i\t' '%i\t' '%i\t' '%i\t'      '%f\t'         '%f\t' % (
            self.seed_instance,
            self.seed_challenges,
            self.n,
            self.k,
            self.N,
            self.result_vote_count,
            self.result_overall_stab,
            self.measured_time
        )

        self.result_logger.info(msg)

    def calculate_stabilities(self, instance, challenge_prng):
        """
        Calculate the stability for random chosen challenges.
        :param instance: SimulationMajorityLTFArray
                         A simulation of a Majority Vote Arbiter PUF.
        :param challenge_prng: RandomState
                               Pseudo-random number generator which is used to generate challenges.
        """
        challenges = tools.random_inputs(self.n, self.N, random_instance=challenge_prng)
        eval_array = np.zeros(len(challenges), dtype=tools.RESULT_TYPE)

        # Evaluation of the PUF in order to measure the stability
        for i in range(self.iterations):
            eval_array = eval_array + instance.eval(challenges)

        # Calculation of the stability for every challenge
        stab_array = (np.abs(eval_array) + self.iterations) / (2 * self.iterations)
        # Number which counts the satisfying challenges
        num_goal_fulfilled = 0
        # Check of the desired_stability
        for i in range(self.N):
            if stab_array[i] >= self.desired_stability:
                num_goal_fulfilled += 1
        # Relative frequency
        self.overall_stab = num_goal_fulfilled / self.N
