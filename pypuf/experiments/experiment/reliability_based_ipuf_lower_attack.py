"""
Learn the lower XOR Arbiter PUF of an iPUF.
"""

from os import getpid
import os.path
from typing import NamedTuple
from uuid import UUID
import pickle
from numpy import logical_and, vstack, sqrt, array, zeros, insert, mean, abs as abs_np, sum as sum_np
from numpy.random.mtrand import RandomState
from scipy.stats import pearsonr
from scipy.special import erfinv

from pypuf.tools import approx_dist, TrainingSet, random_inputs
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.evolution_strategies.reliability_cmaes_learner import ReliabilityBasedCMAES
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF


class Parameters(NamedTuple):
    n: int
    k_up: int
    k_down: int
    seed: int
    noisiness: float
    N: int
    R: int
    abort_delta: float


class Result(NamedTuple):
    experiment_id: UUID
    measured_time: float
    pid: int
    accuracy: float
    iterations: int
    stops: str
    max_possible_acc: float
    cross_model_correlation_lower: list
    cross_model_correlation_upper: list
    discard_count: dict
    iteration_count: dict


class ExperimentReliabilityBasedLowerIPUFLearning(Experiment):
    """
        This class implements an experiment for executing the reliability based CMAES
        learner for XOR LTF arrays.
        Furthermore, the learning results are being logged into csv files.
    """

    def __init__(self, progress_log_name, parameters: Parameters):
        """
            Initialize an Experiment using the Reliability based CMAES Learner for
            modeling LTF Arrays.
            :param progress_log_name:   Log name, Prefix of the name of the experiment log
                                        file
            :param parameters:          Parameters object for this experiment
        """

        super().__init__(
            progress_log_name=f'{progress_log_name}.0x{parameters.seed}_{parameters.k_up}_'
                              f'{parameters.k_down}_{parameters.n}_{parameters.N}_{parameters.R}',
            parameters=parameters,
        )
        self.prng = RandomState(seed=self.parameters.seed)
        self.simulation = None
        self.learner = None
        self.model = None
        self.ts = None
        self.learning_meta_data = None
        self.challenges = None
        self.responses = None
        self.s = None
        self.s_swap = None

    def generate_unreliable_challenges_for_lower_layer(self):
        """Analysis outside of attacker model"""
        # chosen_total = sum(~self.s & ~self.s_swap)
        # responses = self.eval_repeat(self.simulation.down, self.challenges[~self.s & ~self.s_swap], self.parameters.R)
        # chosen_unrel = sum(~self.is_reliable(responses))
        # print(f'Out of {chosen_total} chosen challenges, {chosen_unrel} ({chosen_unrel/chosen_total*100:.2f}%) '
        #       f'are actually unreliable on the upper layer.')
        return self.challenges[~self.s & ~self.s_swap], self.responses[~self.s & ~self.s_swap]

    def generate_reliable_challenges_for_lower_layer(self):
        """Analysis outside of attacker model"""
        # chosen_total = sum(self.s & self.s_swap)
        # responses = self.eval_repeat(self.simulation.down, self.challenges[self.s & self.s_swap], self.parameters.R)
        # chosen_rel = sum(self.is_reliable(responses))
        # print(f'Out of {chosen_total} chosen challenges, {chosen_rel} ({chosen_rel/chosen_total*100:.2f}%) '
        #       f'are actually reliable on the upper layer.')
        return self.challenges[self.s & self.s_swap], self.responses[self.s & self.s_swap]

    def run(self):
        """
            Initialize the instance, the training set and the learner
            to then run the Reliability based CMAES with the given parameters.
        """

        # Instantiate the baseline Noisy iPUF from which the lower chains shall be learned
        self.simulation = InterposePUF(
            n=self.parameters.n,
            k_up=self.parameters.k_up,
            k_down=self.parameters.k_down,
            transform='atf',
            seed=self.prng.randint(2**32),
            noisiness=self.parameters.noisiness,
            noise_seed=self.prng.randint(2**32),
        )
        self.ts = TrainingSet(self.simulation, 1, self.prng, self.parameters.R)

        # Caching
        trainset_cache_fn = '/tmp/trainset.cache'
        if os.path.exists(trainset_cache_fn) and False:
            print('WARNING: USING CACHED TRAINING SET!')
            with open(trainset_cache_fn, 'rb') as f:
                self.ts = pickle.load(f)
        else:
            self.generate_crp_set(self.parameters.N)
            # Build training Set for learning the lower chains of the IPUF
            cs_unreliable, rs_unreliable = self.generate_unreliable_challenges_for_lower_layer()
            cs_reliable, rs_reliable = self.generate_reliable_challenges_for_lower_layer()
            cs_train = vstack((cs_unreliable, cs_reliable))
            rs_train = vstack((rs_unreliable, rs_reliable))
            cs_train_p = insert(cs_train, self.simulation.interpose_pos, 1, axis=1)
            cs_train_m = insert(cs_train, self.simulation.interpose_pos, -1, axis=1)
            cs_train = vstack((cs_train_p, cs_train_m))
            rs_train = vstack((rs_train, rs_train))
            # self.ipuf.down.weight_array = np.delete(self.ipuf.down.weight_array, self.ipuf.interpose_pos, axis=1)
            self.ts.instance = self.simulation
            self.ts.challenges = cs_train
            self.ts.responses = rs_train
            self.ts.N = cs_train.shape[0]
            print(f'Generated Training Set: Reliables: {cs_reliable.shape[0]}'
                  f'Unreliables (lower): {cs_unreliable.shape[0]} TrainSetSize: {cs_train.shape[0]}')
            with open(trainset_cache_fn, 'wb+') as f:
                pickle.dump(self.ts, f)

        # Instantiate the CMA-ES learner
        self.learner = ReliabilityBasedCMAES(
            training_set=self.ts,
            k=self.parameters.k_down,
            n=self.parameters.n + 1,
            transform=self.simulation.down.transform,
            combiner=self.simulation.down.combiner,
            abort_delta=self.parameters.abort_delta,
            random_seed=self.prng.randint(2**32) + 1,
            logger=self.progress_logger,
            gpu_id=self.gpu_id,
        )

        # Start learning a model
        self.model, self.learning_meta_data = self.learner.learn()

    def analyze(self):
        """
            Analyze the results and return the Results object.
        """
        n = self.parameters.n + 1 + 1

        # Accuracy of the learned model using 10000 random samples.
        empirical_accuracy = 1 - approx_dist(self.simulation.down, self.model, 10000, RandomState(1902380))

        # Accuracy of the base line Noisy LTF. Can be < 1.0 since it is noisy.
        best_empirical_accuracy = 1 - approx_dist(self.simulation.down, self.simulation.down, 10000, RandomState(12346))
        # Correl. of the learned model and the base line LTF using pearson for all chains
        """
        cross_model_correlation = [[pearsonr(v[:n], w[:n])[0]
                                        for w in self.model.weight_array]
                                        for v in self.ts.instance.down.weight_array]
        """
        cross_model_correlation_lower = [[round(pearsonr(v, w)[0], 4)
                                          for w in self.ts.instance.down.weight_array]
                                         for v in self.model.weight_array]
        cross_model_correlation_upper = [[round(pearsonr(v[array(range(66)) != 32], w)[0], 4)
                                          for w in self.ts.instance.up.weight_array]
                                         for v in self.model.weight_array]

        return Result(
            experiment_id=self.id,
            measured_time=self.measured_time,
            pid=getpid(),
            accuracy=empirical_accuracy,
            iterations=self.learner.num_iterations,
            stops=self.learner.stops,
            max_possible_acc=best_empirical_accuracy,
            cross_model_correlation_lower=cross_model_correlation_lower,
            cross_model_correlation_upper=cross_model_correlation_upper,
            discard_count=self.learning_meta_data['discard_count'],
            iteration_count=self.learning_meta_data['iteration_count']
        )

    def eval_repeat(self, simulation, challenges, repetitions=None):
        if repetitions is None:
            repetitions = self.parameters.R
        responses = zeros((challenges.shape[0], repetitions))
        for i in range(repetitions):
            responses[:, i] = simulation.eval(challenges[:, :]).T
        return responses

    def interpose(self, challenges, bit):
        return insert(challenges, self.simulation.interpose_pos, bit, axis=1)

    @staticmethod
    def is_reliable(responses, epsilon=0.8):
        return abs_np(mean(responses, axis=1)) >= epsilon  # TODO verify eps implementation

    def generate_crp_set(self, N):
        cs = random_inputs(self.parameters.n, N, self.prng)
        cs_swap = cs.copy()
        cs_swap[:, self.simulation.interpose_pos // 2 - 1] *= -1
        cs_swap[:, self.simulation.interpose_pos // 2] *= -1

        rs = self.eval_repeat(self.simulation, cs)
        s = self.is_reliable(rs)
        s_swap = self.is_reliable(self.eval_repeat(self.simulation, cs_swap))
        l_p = self.is_reliable(self.eval_repeat(self.simulation.down, self.interpose(cs, 1)))
        l_p_swap = self.is_reliable(self.eval_repeat(self.simulation.down, self.interpose(cs_swap, 1)))
        l_m = self.is_reliable(self.eval_repeat(self.simulation.down, self.interpose(cs, -1)))
        l_m_swap = self.is_reliable(self.eval_repeat(self.simulation.down, self.interpose(cs_swap, -1)))

        for label, event, condition in [
            ('l_p|s', l_p, s),
            ('l_p|s_swap', l_p, s_swap),
            ('l_p|11', l_p, s & s_swap),
            ('l_p|10', l_p, s & ~s_swap),
            ('l_p|01', l_p, ~s & s_swap),
            ('l_p|00', l_p, ~s & ~s_swap),
            ('l_p_swap|s', l_p_swap, s),
            ('l_p_swap|s_swap', l_p_swap, s_swap),
            ('l_p_swap|11', l_p_swap, s & s_swap),
            ('l_p_swap|10', l_p_swap, s & ~s_swap),
            ('l_p_swap|01', l_p_swap, ~s & s_swap),
            ('l_p_swap|00', l_p_swap, ~s & ~s_swap),
            ('l_m|s', l_m, s),
            ('l_m|s_swap', l_m, s_swap),
            ('l_m|11', l_m, s & s_swap),
            ('l_m|10', l_m, s & ~s_swap),
            ('l_m|01', l_m, ~s & s_swap),
            ('l_m|00', l_m, ~s & ~s_swap),
            ('l_m_swap|s', l_m_swap, s),
            ('l_m_swap|s_swap', l_m_swap, s_swap),
            ('l_m_swap|11', l_m_swap, s & s_swap),
            ('l_m_swap|10', l_m_swap, s & ~s_swap),
            ('l_m_swap|01', l_m_swap, ~s & s_swap),
            ('l_m_swap|00', l_m_swap, ~s & ~s_swap),
        ]:
            count_condition = sum_np(condition)
            coincidences = sum_np(logical_and(event, condition))
            print(f'{label}: {coincidences/count_condition:.4f} (total: {coincidences})')

        self.challenges = cs
        self.responses = rs
        self.s = s
        self.s_swap = s_swap
