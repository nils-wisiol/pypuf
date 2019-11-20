"""
Learn the lower XOR Arbiter PUF of an IPUF.
"""

from os import getpid
from typing import NamedTuple
from uuid import UUID

import numpy as np
from numpy.random.mtrand import RandomState
from scipy.stats import pearsonr

from pypuf.tools import approx_dist, TrainingSet
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.evolution_strategies.reliability_cmaes_learner import ReliabilityBasedCMAES
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF


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
    cross_model_correlation: list
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

    def generate_unreliable_challenges_for_lower_puf(self):
        print("Generating unreliable challenges for the lower PUF.")
        ts = TrainingSet(self.ipuf,
                         self.parameters.num,
                         random_instance=self.parameters.prng,
                         reps=self.parameters.reps)
        resps = np.int8(ts.responses)
        # Find unreliable challenges on whole IPUF
        unrels = np.abs(np.mean(resps, axis=1)) < 0.5
        unrel_chals = training_set.challenges[unrels]
        unrel_resps = training_set.responses[unrels]
        """print("Found %d unreliable challenges." % np.sum(unrels))"""
        # Flip middle bits of these challenges
        flipped_chals = unrel_chals.copy()
        flipped_chals[:,self.n//2 - 1] *= -1
        flipped_chals[:,self.n//2] *= -1
        flipped_resps = np.zeros(resps[unrels].shape)
        # Find reliable challenges among these flipped challenges
        for i in range(nReps):
            flipped_resps[:, i] = ipuf.eval(flipped_chals).T
        flipped_rels = np.abs(np.mean(flipped_resps, axis=1)) > 0.5
        candidate_chals = unrel_chals[flipped_rels]
        candidate_resps = unrel_resps[flipped_rels]
        """print("-> Among those, %d are reliable when c is flipped." % np.sum(flipped_rels))"""
        return candidate_chals, candidate_resps

    def generate_reliable_challenges_for_IPUF(self, int: num_chals):
        chal_count = 0
        candidate_chals = None
        candidate_resps = None
        while chal_count <= num_chals:
            ts = TrainingSet(self.ipuf,
                             self.parameters.num,
                             random_instance=self.parameters.prng,
                             reps=self.parameters.reps)
            resps = np.int8(ts.responses)
            # Find reliable challenges on whole IPUF
            rels = np.abs(np.mean(resps, axis=1)) > 0.5
            rel_chals = ts.challenges[rels]
            rel_resps = ts.responses[rels]
            candidate_chals = np.vstack(candidate_chals, rel_chals[:num_chals-chal_count,:])
            candidate_resps = np.vstack(candidate_resps, rel_resps[:num_chals-chal_count,:])
            chal_count += rel_chals.shape[0]
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
                                 seed=self.prng.randint(),
                                 noisiness=self.parameters.noisiness,
                                 noise_seed=self.prng.randint()
                                 )
        self.instance = self.ipuf.down

        # Build training Set for learning the lower chains of the IPUF
        unrel_chals, unrel_resps = self.generate_unreliable_challenges_for_lower_puf()
        rel_size = unrel_chals.shape[0] * 4
        rel_chals, rel_resps = self.generate_reliable_challenges_for_IPUF(rel_size)
        training_chals = np.vstack(unrel_chals, rel_chals)
        training_resps = np.vstack(unrel_resps, rel_resps)
        # -> Insert constant bit (-1) where usually the UP_PUF is injected
        training_chals = np.insert(training_chals, 32, -1, axis=1)
        # Hacky: create TrainingSet and then change the member variables
        self.ts = TrainingSet(self.instance, 1, self.prng, self.parameters.reps)
        self.ts.instance = self.ipuf.down
        self.ts.challenges = training_chals
        self.ts.responses = training_resps
        self.ts.N = training_chals.shape[0]

        # Instantiate the CMA-ES learner
        self.learner = ReliabilityBasedCMAES(
                           self.ts,
                           self.parameters.k,
                           self.parameters.n+1,
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
        n = self.parameters.n + 1

        # Accuracy of the learned model using 10000 random samples.
        empirical_accuracy     = 1 - approx_dist(self.instance, self.model,
                                    10000, RandomState(1902380))

        # Accuracy of the base line Noisy LTF. Can be < 1.0 since it is Noisy.
        best_empirical_accuracy = 1 - approx_dist(self.instance, self.instance,
                                    10000, RandomState(12346))
        # Correl. of the learned model and the base line LTF using pearson for all chains
        cross_model_correlation = [[pearsonr(v[:n], w[:n])[0]
                                        for w in self.model.weight_array]
                                        for v in self.training_set.instance.weight_array]

        return Result(
            experiment_id=self.id,
            measured_time=self.measured_time,
            pid=getpid(),
            accuracy=empirical_accuracy,
            iterations=self.learner.num_iterations,
            stops=self.learner.stops,
            max_possible_acc=best_empirical_accuracy,
            cross_model_correlation=cross_model_correlation,
            discard_count=self.learning_meta_data['discard_count'],
            iteration_count=self.learning_meta_data['iteration_count']
        )
