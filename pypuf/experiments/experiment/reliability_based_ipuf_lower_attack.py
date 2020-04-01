"""
Learn the lower XOR Arbiter PUF of an IPUF.
"""

from os import getpid
from typing import NamedTuple
from uuid import UUID

import numpy as np
from numpy.random.mtrand import RandomState
from scipy.stats import pearsonr

from pypuf.tools import approx_dist, TrainingSet, random_inputs
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.evolution_strategies.reliability_cmaes_learner import ReliabilityBasedCMAES
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF
from pypuf.simulation.arbiter_based.ltfarray import LTFArray

from scipy.special import erfinv

import pickle
import os.path

class Parameters(NamedTuple):
    n: int
    k: int
    seed: int
    noisiness: float
    num: int
    reps: int
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


def isUnreliable(responses):
    return np.abs(np.mean(responses, axis=1)) < 0.8

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
            '%s.0x%x_%i_%i_%i_%i' % (
                progress_log_name,
                parameters.seed,
                parameters.k,
                parameters.n,
                parameters.num,
                parameters.reps),
            parameters)
        self.prng = RandomState(seed=self.parameters.seed)
        self.training_set = None
        self.instance = None
        self.learner = None
        self.model = None

    def generate_unreliable_challenges_for_upper_puf(self):
        print("Generating unreliable challenges for the upper PUF.")
        ts = TrainingSet(self.ipuf,
                         self.parameters.num,
                         random_instance=self.prng,
                         reps=self.parameters.reps)
        resps = np.int8(ts.responses)
        # Find unreliable challenges on whole IPUF
        unrels = isUnreliable(resps)
        unrel_chals = ts.challenges[unrels]
        unrel_resps = ts.responses[unrels]
        """print("Found %d unreliable challenges." % np.sum(unrels))"""
        # Flip middle bits of these challenges
        flipped_chals = unrel_chals.copy()
        flipped_chals[:,self.parameters.n//2 - 1] *= -1
        flipped_chals[:,self.parameters.n//2] *= -1
        flipped_resps = np.zeros(resps[unrels].shape)
        # Find reliable challenges among these flipped challenges
        for i in range(self.parameters.reps):
            flipped_resps[:, i] = self.ipuf.eval(flipped_chals).T
        flipped_rels = ~isUnreliable(flipped_resps)
        candidate_chals = unrel_chals[flipped_rels]
        candidate_resps = unrel_resps[flipped_rels]
        """print("-> Among those, %d are reliable when c is flipped." % np.sum(flipped_rels))"""

        """Analysis outside of attacker model"""
        chosen_total = sum(flipped_rels)
        responses = np.zeros((chosen_total, self.parameters.reps))
        for i in range(self.parameters.reps):
            responses[:, i] = self.ipuf.up.eval(candidate_chals[:, :]).T
        chosen_unrel = sum(isUnreliable(responses))
        print(f'Out of {chosen_total} chosen challenges, {chosen_unrel} ({chosen_unrel/chosen_total*100:.2f}%) '
              f'are actually unreliable on the upper layer.')

        return candidate_chals, candidate_resps

    def generate_reliable_challenges_for_upper_puf(self):
        print("Generating reliable challenges for the upper PUF.")
        ts = TrainingSet(self.ipuf,
                         self.parameters.num,
                         random_instance=self.prng,
                         reps=self.parameters.reps)
        resps = np.int8(ts.responses)
        # Find reliable challenges on whole IPUF
        rels = ~isUnreliable(resps)
        rel_chals = ts.challenges[rels]
        rel_resps = ts.responses[rels]
        """print("Found %d reliable challenges." % np.sum(unrels))"""
        # Flip middle bits of these challenges
        flipped_chals = rel_chals.copy()
        flipped_chals[:,self.parameters.n//2 - 1] *= -1
        flipped_chals[:,self.parameters.n//2] *= -1
        flipped_resps = np.zeros(resps[rels].shape)
        # Find unreliable challenges among these flipped challenges
        for i in range(self.parameters.reps):
            flipped_resps[:, i] = self.ipuf.eval(flipped_chals).T
        flipped_unrels = isUnreliable(flipped_resps)
        candidate_chals = rel_chals[flipped_unrels]
        candidate_resps = rel_resps[flipped_unrels]
        """print("-> Among those, %d are reliable when c is flipped." % np.sum(flipped_rels))"""

        """Analysis outside of attacker model"""
        chosen_total = sum(flipped_unrels)
        responses = np.zeros((chosen_total, self.parameters.reps))
        for i in range(self.parameters.reps):
            responses[:, i] = self.ipuf.up.eval(candidate_chals[:, :]).T
        chosen_rel = sum(~isUnreliable(responses))
        print(f'Out of {chosen_total} chosen challenges, {chosen_rel} ({chosen_rel/chosen_total*100:.2f}%) '
              f'are actually reliable on the upper layer.')

        return candidate_chals, candidate_resps

    def run(self):
        """
            Initialize the instance, the training set and the learner
            to then run the Reliability based CMAES with the given parameters.
        """

        # Instantiate the baseline Noisy IPUF from which the lower chains shall be learned
        self.ipuf = InterposePUF(n=self.parameters.n,
                                 k_up=1,
                                 k_down=self.parameters.k,
                                 transform='atf',
                                 seed=self.prng.randint(2**32),
                                 noisiness=self.parameters.noisiness,
                                 noise_seed=self.prng.randint(2**32)
                                 )
        self.instance = self.ipuf.down

        # Caching
        trainset_cache_fn = '/tmp/trainset.cache'
        if os.path.exists(trainset_cache_fn) and False:
            print('WARNING: USING CACHED TRAINING SET!')
            with open(trainset_cache_fn, 'rb') as f:
                self.ts = pickle.load(f)
        else:
            self.generate_challenge_statistics(self.parameters.num)
            # Build training Set for learning the lower chains of the IPUF
            unrel_chals, unrel_resps = self.generate_unreliable_challenges_for_upper_puf()
            rel_chals, rel_resps = self.generate_reliable_challenges_for_upper_puf()
            training_chals = np.vstack((unrel_chals, rel_chals))
            training_resps = np.vstack((unrel_resps, rel_resps))
            # Hacky: create TrainingSet and then change the member variables
            self.ts = TrainingSet(self.instance, 1, self.prng, self.parameters.reps)
            #self.ipuf.down.weight_array = np.delete(self.ipuf.down.weight_array, self.ipuf.interpose_pos, axis=1)
            self.ts.instance = self.ipuf
            self.ts.challenges = training_chals
            self.ts.responses = training_resps
            self.ts.N = training_chals.shape[0]
            print("Generated Training Set: Reliables: %d Unreliables (lower): %d TrainSetSize: %d"
                  % (rel_chals.shape[0], unrel_chals.shape[0], training_chals.shape[0]))
            with open(trainset_cache_fn, 'wb+') as f:
                pickle.dump(self.ts, f)


        # DEBUGGING: Unique challenges
        lin_chals = self.instance.transform(self.ts.challenges,
                                            k=self.parameters.k)
        W_down = np.array(self.ts.instance.up.weight_array[:, :-1])
        delay_diffs = W_down.dot(lin_chals.T[:, 0, :])
        thresh = np.sqrt(2) * 0.05 * erfinv(2*0.7-1)
        uc = np.abs(delay_diffs) < thresh
        overlaps = np.sum(uc, axis=0) > 1
        print("Unreliable Challenges", np.sum(uc, axis=1))
        print("Overlapping Unreliable Challenges", np.sum(uc[:, overlaps], axis=1))

        """
        W_down = np.array(self.ts.instance.down.weight_array[:,:-1])
        delay_diffs = W_down.dot(self.ts.challenges.T)
        from scipy.special import erfinv
        thresh = np.sqrt(2) * 0.05 * erfinv(2*0.8-1)
        unreliable_chals = np.abs(delay_diffs) < 0.000005
        print("ASDASd",np.sum(unreliable_chals, axis=1))

        self.ts = TrainingSet(self.ipuf, self.parameters.num, self.prng, self.parameters.reps)
        self.ts.instance = self.ipuf.down
        self.ts.challenges = np.insert(self.ts.challenges, self.ipuf.interpose_pos, 1, axis=1)
        """

        # Instantiate the CMA-ES learner
        self.learner = ReliabilityBasedCMAES(
                           self.ts,
                           1,
                           self.parameters.n,
                           self.instance.transform,
                           self.instance.combiner,
                           self.parameters.abort_delta,
                           self.prng.randint(2**32),
                           self.progress_logger,
                           self.gpu_id)

        # Start learning a model
        self.model, self.learning_meta_data = self.learner.learn()


    def analyze(self):
        """
            Analyze the results and return the Results object.
        """
        n = self.parameters.n + 1 + 1

        # Accuracy of the learned model using 10000 random samples.
        empirical_accuracy     = 1 - approx_dist(self.ipuf.up,
                                    self.model,
                                    10000, RandomState(1902380))

        # Accuracy of the base line Noisy LTF. Can be < 1.0 since it is Noisy.
        best_empirical_accuracy = 1 - approx_dist(self.instance, self.instance,
                                    10000, RandomState(12346))
        # Correl. of the learned model and the base line LTF using pearson for all chains
        """
        cross_model_correlation = [[pearsonr(v[:n], w[:n])[0]
                                        for w in self.model.weight_array]
                                        for v in self.ts.instance.down.weight_array]
        """
        cross_model_correlation_lower = [[]]
        cross_model_correlation_upper = [[round(pearsonr(v, w)[0], 4)
                                        for w in self.ts.instance.up.weight_array]
                                        for v in self.model.weight_array]


        print(np.array(self.ts.instance.up.weight_array[:,:-1]).shape)
        print(self.ts.challenges.T.shape)
        W_down = np.array(self.ts.instance.up.weight_array[:,:-1])
        delay_diffs = W_down.dot(self.ts.challenges.T)
        from scipy.special import erfinv
        thresh = np.sqrt(2) * 0.05 * erfinv(2*0.8-1)
        unreliable_chals = delay_diffs < thresh
        print(np.sum(unreliable_chals, axis=1))

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

    def eval_repeat(self, instance, challenges, reps=None):
        if reps is None:
            reps = self.parameters.reps
        responses = np.zeros((challenges.shape[0], reps))
        for i in range(reps):
            responses[:, i] = instance.eval(challenges[:, :]).T
        return responses

    def interpose(self, challenges, bit):
        return np.insert(challenges, self.ipuf.interpose_pos, bit, axis=1)

    def is_reliable(self, responses, eps=.8):
        return np.abs(np.mean(responses, axis=1)) >= eps  # TODO verify eps implementation

    def generate_challenge_statistics(self, N):
        challenges = random_inputs(self.parameters.n, N, RandomState(1))
        challenges_swapped = challenges.copy()
        challenges_swapped[:, self.ipuf.interpose_pos // 2 - 1] *= -1
        challenges_swapped[:, self.ipuf.interpose_pos // 2] *= -1

        s = self.is_reliable(self.eval_repeat(self.ipuf, challenges))
        s_swap = self.is_reliable(self.eval_repeat(self.ipuf, challenges_swapped))
        u = self.is_reliable(self.eval_repeat(self.ipuf.up, challenges))
        # u_swap = self.is_reliable(self.eval_repeat(self.ipuf.up, challenges_swapped))
        # l_p = self.eval_repeat(self.ipuf.down, self.interpose(challenges, 1))
        # l_p_swap = self.eval_repeat(self.ipuf.down, self.interpose(challenges_swapped, 1))
        # l_m = self.eval_repeat(self.ipuf.down, self.interpose(challenges, -1))
        # l_m_swap = self.eval_repeat(self.ipuf.down, self.interpose(challenges_swapped, 1))

        for label, event, condition in [
            ('u|s', u, s),
            ('u|s_swap', u, s_swap),
            ('u|11', u, s & s_swap),
            ('u|10', u, s & ~s_swap),
            ('u|01', u, ~s & s_swap),
            ('u|00', u, ~s & ~s_swap),
        ]:
            pr_condition = sum(condition) / N
            pr_event_and_condition = sum(event & condition) / N
            print(f'{label}: {pr_event_and_condition/pr_condition:.4f}')
