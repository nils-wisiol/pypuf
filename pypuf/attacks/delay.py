import logging
from typing import Callable, List, Optional

import numpy as np
import tensorflow as tf
from cma import CMA
from scipy.stats import pearsonr
from tensorflow.python import erf

from .. import random
from ..io import ChallengeResponseSet, ChallengeReliabilitySet
from ..metrics.common import approx_accuracy, approx_similarity_data
from ..simulation.base import Simulation
from ..simulation.delay import LTFArray, ArbiterPUF


class EarlyStop(BaseException):

    def __init__(self, reasons: List[str], *args: object) -> None:
        super().__init__(*args)
        self.reasons = reasons


class GapChainAttack:

    @classmethod
    def _pearsonr(cls, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Calculates Pearson Correlation Coefficient.
        x and y are matrices with data vectors as columns.
        Return array where index i,j is the pearson correlation of i'th
        column vector in x and the j'th column vector in y.
        """
        x_centered = x - tf.reduce_mean(x, axis=0)
        y_centered = y - tf.reduce_mean(y, axis=0)
        return cls._pearsonr_centered(x_centered, y_centered)

    @staticmethod
    def _pearsonr_centered(x_centered: tf.Tensor, y_centered: tf.Tensor) -> tf.Tensor:
        cov_xy = tf.tensordot(tf.transpose(x_centered), y_centered, axes=1)
        auto_cov = tf.sqrt(tf.tensordot(
            tf.reduce_sum(tf.square(x_centered), axis=0),
            tf.reduce_sum(tf.square(y_centered), axis=0),
            axes=0,
        ))
        corr = cov_xy / auto_cov
        return corr

    def _reliability_pearson(self, x: tf.Tensor) -> tf.Tensor:
        # TODO make sure this is correct
        centered_x = x - tf.reduce_mean(x, axis=0)
        return self._pearsonr_centered(self._puf_reliabilities_centered, centered_x)

    def _init_learner_state(self, seed: int) -> np.ndarray:
        state = np.empty(shape=(self.n + 1))
        state[:-1] = random.prng(f'GapAttack initial state for seed {seed}').normal(0, 1, size=self.n)  # init weights
        state[-1] = 2  # init eps
        return state

    @staticmethod
    def reliability_model(delay_diffs: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
        return tf.math.greater(tf.transpose(tf.abs(delay_diffs)), eps)

    @staticmethod
    def reliability_crp_set(crps: ChallengeReliabilitySet) -> tf.Tensor:
        r"""
        How to use the reliabilities given. This is actually part of the fitness function,
        but separated as precomputation for performance.

        This implementation returns as reliability for any given challenge ::math`c`

        ..math:: \left| \frac{l}{2} - \sum_{i=1}^l r_i \right|,

        where the math::`r_i` are the `l` responses recorded to challenge ::math`c`, in 0/1 notation,
        following the definition by Becker [Bec15]_.

        As the argument to this function is a ChallengeReliabilitySet, containing the average response
        to each challenge in -1/1 notation, the implementation differs from above formula.
        """
        N = crps.reliabilities.shape[0]
        return 5 * np.absolute(crps.reliabilities).reshape((N,)) / 2  # TODO fix constant r=5

    def __init__(self, crps: ChallengeReliabilitySet, avoid_responses: List[np.ndarray], seed: int,
                 cma_kwargs: Optional[dict], sigma: float, pop_size: int,
                 abort_delta: float,
                 abort_iter: int, stop_early: bool, minimum_fitness_by_sigma: dict, gap_attack) -> None:
        # TODO add gap_attack type annotation GapAttack when upgrading to Python 3.8
        # parameters
        self.n = crps.challenge_length
        self.crps = crps
        self.challenges = np.array(crps.challenges, dtype=np.float64)
        self.avoid_responses = avoid_responses
        self.seed = seed
        self.cma_kwargs = cma_kwargs
        self.sigma = sigma
        self.pop_size = pop_size
        self.abort_delta = abort_delta
        self.abort_iter = abort_iter
        self.stop_early = stop_early
        self.minimum_fitness_by_sigma = minimum_fitness_by_sigma
        self.gap_attack: GapAttack = gap_attack

        # histories
        self.fitness_history = []
        self.sigma_history = []
        self.avoid_responses_score_history = []

        # learner state
        self.cma = None
        self.current_result = None
        self.current_fitness = None
        self.current_responses = None
        self._callback_generation = None

        # learner result
        self.result = None
        self.weights = None
        self.eps = None
        self.fitness = None
        self.generations = None
        self.early_stop_reasons = None
        self.abort_reasons = None

        # precomputation
        puf_reliabilities = self.reliability_crp_set(self.crps)
        self._puf_reliabilities_centered = puf_reliabilities - tf.reduce_mean(puf_reliabilities)

    def objective(self, state: tf.Tensor) -> tf.Tensor:
        # Weights and epsilon have the first dim as number of population
        weights = state[:, :self.n]
        eps = state[:, -1]

        # compute model reliabilities
        delay_diffs = tf.linalg.matmul(weights, self.challenges.T)
        model_reliabilities = self.reliability_model(delay_diffs, eps)

        # compute pearson correlation coefficients with target reliabilities
        x = tf.cast(model_reliabilities, tf.double)
        fitness = tf.abs(self._reliability_pearson(x) - tf.constant(1, dtype=tf.float64))

        # record fitness and current responses
        best = tf.argmin(fitness)
        # self.fitness_history.append(fitness[best].numpy())  # TODO WRONG
        self.current_responses = tf.sign(delay_diffs[best])

        return fitness

    def early_stop(self, cma: CMA) -> List[str]:
        """
        Determines if the learning should be aborted early by returning a list of reasons why to abort. Returning ``[]``
        continues the learning process.

        We abort early if the response behavior of the currently learned chain is similar to already known chains, or
        if the evolution of the sigma and fitness values is typical for an unsuccessful run. This follows [Bec15]_:

            In general, unsuccessful runs can be aborted early to greatly decrease the computation time of the attack.
            To determine which runs are likely to be unsuccessful, the global mutation parameter 𝜎 in conjunction with
            the fitness value can be used. Furthermore, the hamming distance between responses from the model under
            test and the already computed PUF models can be used to detect runs that are converging to a PUF model
            that has already been found. These runs can also be aborted early to considerably speed up the computation
            time.

        This function is called once per generation and is passed the CMA learner object.
        """
        reasons = []

        # check if response behavior is similar to any of those that should be avoided
        if self.current_responses is not None and self.avoid_responses:
            # compute a similarity score of the current fittest chain to each known chain
            scores = [
                3 * np.absolute(.5 - np.average(approx_similarity_data(r, self.current_responses.numpy())))
                for r in self.avoid_responses
            ]
            self.avoid_responses_score_history.append(scores)
            score = max(scores)
            if .3 < cma.σ.numpy() < score:  # this is where the magic happens
                reasons.append(f'similar to already accepted chain with similarity score {score:.4f} '
                               f'at sigma {cma.σ.numpy():.4f}')

        # check if fitness is not good enough for current sigma value
        if self.fitness_history:
            current_fitness = self.fitness_history[-1]
            for sigma, min_fitness in self.minimum_fitness_by_sigma.items():
                if cma.σ < sigma and current_fitness > min_fitness:
                    reasons.append(f'low sigma high fitness sigma={cma.σ.numpy():.4f} fitness={current_fitness:.4f}')

        return reasons

    def prepare(self) -> None:
        # for reproducible results
        tf.random.set_seed(random.seed(f'GapAttack tensorflow seed for seed {self.seed}'))
        init_state = self._init_learner_state(self.seed)
        self._callback_generation = 0

        # prepare callback function, will be called after each generation
        def callback_hook(_cma: CMA, logger: logging.Logger) -> None:
            # only one callback per generation
            if _cma.generation == self._callback_generation:
                return
            self._callback_generation = _cma.generation

            # record data
            self.sigma_history.append(_cma.σ.numpy())
            self.fitness_history.append(_cma.best_fitness())

            # stop early
            early_stop_reasons = self.early_stop(_cma)
            if early_stop_reasons and self.stop_early:
                raise EarlyStop(early_stop_reasons)

        # initialize learner
        self.cma = CMA(
            initial_solution=init_state,
            initial_step_size=self.sigma,
            fitness_function=self.objective,
            termination_no_effect=self.abort_delta,
            population_size=self.pop_size,
            callback_function=callback_hook,
            **(self.cma_kwargs or {}),
        )

    def step(self) -> bool:
        try:
            self.current_result, self.current_fitness = self.cma.search(max_generations=1)
            if self.cma.termination_criterion_met or self.cma.generation == self.abort_iter:
                self.early_stop_reasons = []
                return False
            else:
                return True
        except EarlyStop as e:
            self.current_result, self.current_fitness = self.cma.best_solution(), self.cma.best_fitness()
            self.early_stop_reasons = e.reasons
            return False

    def finish(self) -> None:
        self.result = self.current_result
        self.weights, self.eps = self.result[:self.n], self.result[-1]
        self.fitness = self.current_fitness
        self.abort_reasons = [key for key, val in self.cma.should_terminate(True)[1].items() if val]
        if self.cma.generation == self.abort_iter:
            self.abort_reasons.append('max generations')


class GapChainAttackContinuous(GapChainAttack):

    @staticmethod
    def reliability_model(delay_diffs: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
        return erf(tf.transpose(tf.abs(delay_diffs)))

    @staticmethod
    def reliability_crp_set(crps: ChallengeReliabilitySet) -> tf.Tensor:
        N = crps.reliabilities.shape[0]
        return np.absolute(crps.reliabilities).reshape((N,)).astype(np.float64)

    def _init_learner_state(self, seed: int) -> np.ndarray:
        return random.prng(f'GapAttack initial state for seed {seed}').normal(0, 1, size=self.n)  # init weights


class GapChainAttackPenalty(GapChainAttack):

    AVOID_WEIGHTS_MASK_CORR = .6

    def __init__(self, crps: ChallengeReliabilitySet, avoid_responses: List[np.ndarray], seed: int,
                 cma_kwargs: Optional[dict], sigma: float, pop_size: int, abort_delta: float, abort_iter: int,
                 stop_early: bool, minimum_fitness_by_sigma: dict, gap_attack) -> None:
        super().__init__(crps, avoid_responses, seed, cma_kwargs, sigma, pop_size, abort_delta, abort_iter, stop_early,
                         minimum_fitness_by_sigma, gap_attack)
        self.avoid_weights = np.array(self.gap_attack.results + self.gap_attack.results_discarded)
        self.current_penalty = None
        self.current_penalty_corr = None

    def objective(self, state: tf.Tensor) -> tf.Tensor:
        fitness = super().objective(state)

        if self.avoid_weights.shape[0]:
            # compute absolute correlations with all weights that should be avoided
            weights = state[:, :self.n]
            corr = tf.abs(self._pearsonr(self.avoid_weights.T, tf.transpose(weights)))

            # the correlation score is the sum of those
            corr_score = tf.reduce_sum(corr, axis=0)

            # penalty is the excess of the corr score of an individual over the average score
            penalty = tf.maximum(corr_score - tf.reduce_mean(corr_score), 0)

            # only log when we assume the caller is the CMA-ES
            # other calls happen when someone uses best_fitness(), in that case we don't want to record anything
            if state.shape[0] > 1:
                self.current_penalty_corr = tf.reduce_sum(corr, axis=0).numpy()
                self.current_penalty = penalty.numpy()

            return fitness + penalty

        return fitness


class GapAttack:

    def __init__(self,
                 crps: ChallengeReliabilitySet,
                 k_max: int,
                 transform: Callable,
                 discard_threshold: float,
                 pop_size: int = 25,
                 abort_delta: float = 5e-3,
                 abort_iter: int = 500,
                 minimum_fitness_by_sigma: dict = None,
                 sigma: float = 1.0,
                 stop_early: bool = True,
                 chain_attack_type: type(GapChainAttack) = GapChainAttack,
                 ) -> None:
        # parameters
        self.n = crps.challenge_length
        self.crps = crps
        self.k_max = k_max
        self.transform = transform
        self.pop_size = pop_size
        self.abort_delta = abort_delta
        self.abort_iter = abort_iter
        self.minimum_fitness_by_sigma = minimum_fitness_by_sigma
        self.discard_threshold = discard_threshold
        self.sigma = sigma
        self.stop_early = stop_early
        self.chain_attack_type = chain_attack_type

        # learner status
        self.learner = None
        self.results = []
        self.results_discarded: List[np.ndarray] = []
        self.avoid_responses: List[np.ndarray] = []
        self.attempts = 0
        self.traces = []
        self.generations = 0

        # result
        self.model = None

        # sanity checks
        if crps.reliabilities.shape[1] != 1:
            raise ValueError(f'GapAttack only supports 1-bit responses, but reliabilities of shape '
                             f'{crps.reliabilities.shape} were given (expected (N, 1, r)).')

        # precomputations
        self._linearized_challenges = self.transform(self.crps.challenges, k=self.k_max)

    def discard_result(self, learner: GapChainAttack) -> List[str]:
        discard_reasons = []
        weights = learner.result[:self.n]
        if weights[0] < 0:
            weights *= -1  # normalize weights to have positive first weight

        # check if fitness is sufficient (below)
        if learner.fitness >= self.discard_threshold or np.isnan(learner.fitness) or np.isinf(learner.fitness):
            discard_reasons.append('fitness too high')

        # Check if learned model (w) is a 'new' chain (not correlated to other chains)
        for i, v in enumerate(self.results):
            corr = tf.abs(pearsonr(weights, v)[0])
            if corr > .9:
                discard_reasons.append(f'too similar, to learned chain {i} (correlation {corr.numpy():.2f})')

        # check if stopped early
        discard_reasons += learner.early_stop_reasons or []

        return discard_reasons

    def prepare(self, seed: int) -> None:
        seed = random.seed(f'GapAttack on seed {seed} attempt {self.attempts}')
        target_chain = len(self.results)
        logging.debug(f'Attempting to learn {target_chain + 1}-th chain with seed {seed} (attempt {self.attempts})')

        self.learner = self.chain_attack_type(
            crps=ChallengeReliabilitySet(self._linearized_challenges[:, target_chain, :], self.crps.reliabilities),
            avoid_responses=self.avoid_responses,
            seed=seed,
            cma_kwargs=None,
            sigma=self.sigma,
            pop_size=self.pop_size,
            abort_delta=self.abort_delta,
            abort_iter=self.abort_iter,
            stop_early=self.stop_early,
            minimum_fitness_by_sigma=self.minimum_fitness_by_sigma,
            gap_attack=self,
        )
        self.learner.prepare()

    def accept_discard(self) -> List[str]:
        self.attempts += 1
        self.generations += self.learner.cma.generation

        discard_reasons = self.discard_result(self.learner)
        if discard_reasons:
            logging.debug('Discarding chain due to ' + ', '.join(discard_reasons))
            self.results_discarded.append(self.learner.result[:self.n])
            return discard_reasons
        else:
            # accept this chain, record weights and response behavior
            self.results.append(self.learner.result[:self.n])
            self.avoid_responses.append(self.learner.current_responses.numpy())
            logging.debug(f'Adding {len(self.results)}-th chain to pool, '
                          f'{self.k_max - len(self.results)} still missing')
            return []

    def build_model(self) -> LTFArray:
        if self.results:
            # if training accuracy < 0.5, we flip a chain to flip all outputs
            self.model = LTFArray(np.array(self.results)[:, :self.n], self.transform)
            crps = ChallengeResponseSet(self.crps.challenges, np.sign(self.crps.reliabilities))
            if np.average(approx_accuracy(self.model, crps)) < .5:  # TODO
                self.model.weight_array[0] *= -1
        else:
            # if we couldn't find a single chain, just provide a random model to not break the interface
            self.model = ArbiterPUF(
                n=self.n, transform=self.transform,
                seed=random.seed('GapAttack on seed no result guess'),
            )

        return self.model

    def learn(self, seed: int, max_retries: int = np.inf) -> Simulation:
        current_attempt = 0

        # loop until k_max chains found
        while len(self.results) < self.k_max and current_attempt < max_retries:
            self.prepare(seed)

            # loop until termination
            while self.learner.step():
                pass
            self.learner.finish()

            logging.debug(f'Found result with fitness {self.learner.fitness:.3f}, aborting/early stopping '
                          f'due to {", ".join(self.learner.abort_reasons + self.learner.early_stop_reasons)}')

            # see if result is good
            self.accept_discard()

        return self.build_model()
