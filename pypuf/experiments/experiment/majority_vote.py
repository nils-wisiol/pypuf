"""
This module provides an experiment class which implements an experiment in order to find an approximately  minimal
number of votes for a Majority Vote Arbiter PUF which satisfy the chosen desired_stability and
overall_desired_stability.
"""
from typing import NamedTuple, Union
from uuid import UUID
import numpy as np
from numpy.random import RandomState
from pypuf import tools
from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.ltfarray import SimulationMajorityLTFArray, LTFArray, NoisyLTFArray


class Parameters(NamedTuple):
    """
    Parameter for MajorityVote experiment.
    """

    # The number of stages of the PUF.
    n: int

    # The number different LTFArrays of the SimulationMajorityLTFArray.
    k: int

    # The number of challenges which are used to evaluate the PUF.
    challenge_count: int

    # The seed which is used to initialize the pseudo-random number generator
    # which is used to generate the stage weights for the arbiter PUF simulation.
    seed_instance: int

    # The random seed which is used to initialize the pseudo-random number
    # generator which is used to generate the noise for the arbiter PUF simulation.
    seed_instance_noise: int

    # The seed which is used to initialize the pseudo-random number generator which
    # is used to generate challenges.
    seed_challenges: int

    # A function: array of int with shape(N,k,n), int number of PUFs k -> shape(N,k,n)
    # The function transforms input challenges in order to increase resistance against attacks.
    transformation: str

    # A function: array of int with shape(N,k,n) -> array of in with shape(N)
    # The functions combines the outputs of k PUFs to one bit results,
    # in oder to increase resistance against attacks.
    combiner: str

    # Mean (“centre”) of the stage weight distribution of the PUF instance simulation.
    mu: float

    # Standard deviation of the stage weight distribution of the PUF instance simulation.
    sigma: float

    # The noisiness factor which is used to scale sigma_noise. The value sigma_noise
    # is the standard deviation of the noise distribution of the PUF instance simulation.
    sigma_noise_ratio: float

    # The number which is used to decide whether a PUF is stable or not.
    desired_stability: float

    # The relative frequency of challenges which are greater equal than the
    # desired_stability.
    overall_desired_stability: float

    # That is the minimum number of votes which are used to evaluate the SimulationMajorityLTFArray
    # instance.
    minimum_vote_count: int

    # The number of evaluations of the SimulationMajorityLTFArray instance which are used
    # to check the desired_stability.
    iterations: int

    # None, float or a one dimensional list of float with shape (k)
    # This bias value or list of bias values will be appended to the weight_array.
    # Use a single value if you want the same bias for all weight_vectors.
    bias: Union[float, list, None] = None


class Result(NamedTuple):
    """
    Result of MajorityVote experiment.
    """

    experiment_id: UUID
    vote_count: int
    overall_stab: float
    measured_time: float


class ExperimentMajorityVoteFindVotes(Experiment):
    """
    This experiment can be used to find an approximately  minimal number of votes for a Majority Vote Arbiter PUF which
    satisfy the chosen desired_stability and overall_desired_stability.
    """

    def __init__(self, progress_log_prefix, parameters):
        super().__init__(
            progress_log_name='%s.0x%x_0_%i_%i_%i_%s_%s' % (
                progress_log_prefix,
                parameters.seed_instance,
                parameters.challenge_count,
                parameters.k,
                parameters.challenge_count,
                parameters.transformation,
                parameters.combiner,
            ),
            parameters=parameters
        )
        self.sigma_noise = None
        self.minimum_vote_count = parameters.minimum_vote_count
        self.maximum_vote_count = 0  # Upper bound for binary search calculated in run()
        self.vote_count = 0  # That is calculated during run()
        self.result_overall_stab = 0.0
        # This saves the first found over all stability which satisfy the overall_desired_stability
        self.first_result_overall_stab = 0.0
        self.result_vote_count = 0
        self.overall_stab = 0.0  # That is the overall_stab for vote_count calculated during run()

    def prepare(self):
        self.sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(
            self.parameters.challenge_count, 1, self.parameters.sigma_noise_ratio)

    def run(self):
        """
        Searches for the minimal number of votes which satisfy self.desired_stability and
        self.overall_desired_stability. The number of votes for a try is given by the interval between self.bottom and
        self.top.
        """
        # Random number generators
        instance_prng = RandomState(self.parameters.seed_instance)
        noise_prng = RandomState(self.parameters.seed_instance_noise)
        challenge_prng = RandomState(self.parameters.seed_challenges)

        # Weight array for the instance which should be learned
        weight_array = LTFArray.normal_weights(self.parameters.challenge_count, self.parameters.k, self.parameters.mu,
                                               self.parameters.sigma, random_instance=instance_prng)

        if isinstance(self.parameters.bias, type('list')):
            self.parameters.bias = np.reshape(np.array(self.parameters.bias), (self.parameters.k, 1))

        self.vote_count = 0
        # Search upper bound for binary search
        while self.overall_stab < self.parameters.overall_desired_stability:
            self.vote_count = self.vote_count * 2 + 1
            puf_instance = SimulationMajorityLTFArray(weight_array, LTFArray.transform_id,
                                                      LTFArray.combiner_xor, self.sigma_noise,
                                                      random_instance_noise=noise_prng, vote_count=self.vote_count,
                                                      bias=self.parameters.bias)
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
            if self.overall_stab >= self.parameters.overall_desired_stability:
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
        return Result(
            experiment_id=self.id,
            vote_count=self.result_vote_count,
            overall_stab=self.result_overall_stab,
            measured_time=self.measured_time,
        )

    def calculate_stabilities(self, instance, challenge_prng):
        """
        Calculate the stability for random chosen challenges.
        :param instance: SimulationMajorityLTFArray
                         A simulation of a Majority Vote Arbiter PUF.
        :param challenge_prng: RandomState
                               Pseudo-random number generator which is used to generate challenges.
        """
        challenges = tools.random_inputs(self.parameters.challenge_count, self.parameters.challenge_count,
                                         random_instance=challenge_prng)
        eval_array = np.zeros(len(challenges), dtype=tools.BIT_TYPE)

        # Evaluation of the PUF in order to measure the stability
        for i in range(self.parameters.iterations):
            eval_array = eval_array + instance.eval(challenges)

        # Calculation of the stability for every challenge
        stab_array = (np.abs(eval_array) + self.parameters.iterations) / (2 * self.parameters.iterations)
        # Number which counts the satisfying challenges
        num_goal_fulfilled = 0
        # Check of the desired_stability
        for i in range(self.parameters.challenge_count):
            if stab_array[i] >= self.parameters.desired_stability:
                num_goal_fulfilled += 1
        # Relative frequency
        self.overall_stab = num_goal_fulfilled / self.parameters.challenge_count
