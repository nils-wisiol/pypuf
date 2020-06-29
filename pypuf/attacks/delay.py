import logging
from typing import Callable, Tuple, List

import numpy as np
import tensorflow as tf
from cma import CMA
from scipy.stats import pearsonr

from .. import random
from ..io import transform_challenge_11_to_01, ChallengeResponseSet
from ..metrics.common import approx_accuracy, approx_similarity_data
from ..simulation.base import Simulation
from ..simulation.delay import LTFArray, ArbiterPUF


class EarlyStop(BaseException):

    def __init__(self, reasons: List[str], *args: object) -> None:
        super().__init__(*args)
        self.reasons = reasons


class GapAttack:

    def __init__(self, crp_set: ChallengeResponseSet, k_max: int, transform: Callable,
                 pop_size: int = 25,
                 abort_delta: float = 5e-3,
                 abort_iter: int = 500,
                 fitness_threshold: float = .9,
                 pool_threshold: float = .9,
                 sigma: float = 1.0) -> None:
        self.crp_set = crp_set
        self.k_max = k_max
        self.n = crp_set.challenge_length
        self.transform = transform
        self.pop_size = pop_size
        self.abort_delta = abort_delta
        self.abort_iter = abort_iter
        self.fitness_threshold = fitness_threshold
        self.pool_threshold = pool_threshold
        self.sigma = sigma
        self.pool_weights = []
        self.pool_responses = []
        self.pool_response_similarity_hist = None
        self.attempt = 0
        self.traces = []
        self.generations = 0

        self.current_challenges = None
        self.fitness_history = None
        self.sigma_history = None
        self.current_responses = None

        if crp_set.responses.shape[1] != 1:
            raise ValueError(f'GapAttack only supports 1-bit responses, but responses of shape '
                             f'{crp_set.responses.shape} were given (expected (N, 1, r)).')

        # Compute PUF Reliabilities. These remain static throughout the optimization.
        self.puf_reliabilities = self.reliability_crp_set(self.crp_set.responses)
        self.puf_reliabilities_centered = self.puf_reliabilities - tf.reduce_mean(self.puf_reliabilities)

        # Linearize challenges for faster LTF computation (shape=(N,k_max,n))
        self._linearized_challenges = self.transform(self.crp_set.challenges, k=self.k_max)

    @classmethod
    def _pearsonr(cls, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        x_centered = x - tf.reduce_mean(x, axis=0)
        y_centered = y - tf.reduce_mean(y, axis=0)
        return cls._pearsonr_centered(x_centered, y_centered)

    @staticmethod
    def _pearsonr_centered(x_centered: tf.Tensor, y_centered: tf.Tensor) -> tf.Tensor:
        cov_xy = tf.tensordot(x_centered, y_centered, axes=1)
        auto_cov = tf.sqrt(tf.reduce_sum(tf.square(x_centered), axis=0) * tf.reduce_sum(tf.square(y_centered)))
        corr = cov_xy / auto_cov
        return corr

    def _reliability_pearson(self, x: tf.Tensor) -> tf.Tensor:
        centered_x = x - tf.reduce_mean(x, axis=0)
        return self._pearsonr_centered(self.puf_reliabilities_centered, centered_x)

    def _init_learner_state(self, seed: int) -> np.ndarray:
        state = np.empty(shape=(self.n + 1))
        state[:-1] = random.prng(f'GapAttack initial state for seed {seed}').normal(0, 1, size=self.n)  # init weights
        state[-1] = 2  # init eps
        return state

    @staticmethod
    def reliability_model(delay_diffs: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
        return tf.math.greater(tf.transpose(tf.abs(delay_diffs)), eps)

    @staticmethod
    def reliability_crp_set(responses: np.ndarray) -> tf.Tensor:
        # Convert to 0/1 from 1/-1
        N, _, r = responses.shape
        responses = responses.reshape(N, r)
        responses = transform_challenge_11_to_01(responses)
        return tf.constant(np.abs(responses.shape[1] / 2 - np.sum(responses, axis=-1)), dtype=tf.double)

    def objective(self, state: tf.Tensor, challenges: tf.Tensor) -> tf.Tensor:
        # Weights and epsilon have the first dim as number of population
        weights = state[:, :self.n]
        eps = state[:, -1]

        # compute model reliabilities
        delay_diffs = tf.linalg.matmul(weights, challenges.T)
        model_reliabilities = self.reliability_model(delay_diffs, eps)

        # compute pearson correlation coefficients with target reliabilities
        x = tf.cast(model_reliabilities, tf.double)
        fitness = tf.abs(self._reliability_pearson(x) - tf.constant(1, dtype=tf.float64))

        # record fitness
        best = tf.argmin(fitness)
        # self.fitness_history.append(fitness[best].numpy())  # WRONG
        self.current_responses = tf.sign(delay_diffs[best])

        return fitness

    def learn(self, seed: int, max_retries: int = np.inf) -> Simulation:
        current_attempt = 0

        # main learning loop
        while len(self.pool_weights) < self.k_max and current_attempt < max_retries:
            chain_seed = random.seed(f'GapAttack on seed {seed} attempt {self.attempt}')
            logging.debug(f'Attempting to learn {len(self.pool_weights) + 1}-th chain with seed {chain_seed} '
                          f'(attempt {self.attempt})')

            weights, eps, fitness, abort_reasons, early_stop_reasons, generations = \
                self.learn_chain(len(self.pool_weights), chain_seed)
            discard_reasons = self.discard_chain(weights, fitness, self.pool_weights)
            logging.debug(f'Found chain with fitness {fitness:.3f}, aborting/early stopping '
                          f'due to {", ".join(abort_reasons + early_stop_reasons)}')

            self.traces.append({
                'weights': weights,
                'eps': eps,
                'fitness': fitness,
                'fitness_hist': self.fitness_history,
                'sigma_hist': self.sigma_history,
                'response_sim_hist': self.pool_response_similarity_hist,
                'abort_reasons': abort_reasons,
                'discard_reasons': discard_reasons,
                'early_stop_reasons': early_stop_reasons,
                'generations': generations,
            })

            self.attempt += 1
            current_attempt += 1
            self.generations += generations

            if discard_reasons or early_stop_reasons:
                logging.debug('Discarding chain due to ' + ', '.join(discard_reasons + early_stop_reasons))
            else:
                # accept this chain, record weights and response behavior
                self.pool_weights.append(weights)
                self.pool_responses.append(self.current_responses.numpy())
                logging.debug(f'Adding {len(self.pool_weights)}-th chain to pool, '
                              f'{self.k_max - len(self.pool_weights)} still missing')

        if self.pool_weights:
            # if training accuracy < 0.5, we flip a chain to flip all outputs
            model = LTFArray(np.array(self.pool_weights), self.transform)
            if np.average(approx_accuracy(model, self.crp_set)) < .5:
                model.weight_array[0] *= -1
        else:
            # if we couldn't find a single chain, just provide a random model to not break the interface
            model = ArbiterPUF(
                n=self.n, transform=self.transform,
                seed=random.seed(f'GapAttack on seed {seed} no result guess'),
            )

        return model

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
        if self.current_responses is not None and self.pool_responses:
            # compute a similarity score of the current fittest chain to each known chain
            scores = [
                3 * np.absolute(.5 - np.average(approx_similarity_data(r, self.current_responses.numpy())))
                for idx, r in enumerate(self.pool_responses)
            ]
            self.pool_response_similarity_hist.append(scores)
            score = max(scores)
            if .3 < cma.σ.numpy() < score:  # this is where the magic happens
                reasons.append(f'similar to already accepted chain with similarity score {score:.4f} '
                               f'at sigma {cma.σ.numpy():.4f}')
        if self.fitness_history and self.fitness_history[-1] > self.fitness_threshold and cma.σ < .5:
            reasons.append(f'low sigma high fitness sigma={cma.σ.numpy():.4f} fitness={self.fitness_history[-1]:.4f}')
        return reasons

    def learn_chain(self, l: int, seed: int, callback: Callable = None,
                    cma_kwargs: dict = None) -> Tuple[np.ndarray, float, float, List[str], List[str], int]:

        # prepare specialized objective function
        self.current_challenges = np.array(self._linearized_challenges[:, l, :], dtype=np.float64)
        self.fitness_history = []
        self.sigma_history = []
        self.pool_response_similarity_hist = []
        self.current_responses = None

        def objective(state: tf.Tensor) -> tf.Tensor:
            return self.objective(state, self.current_challenges)

        # seed tensorflow for reproducible results
        tf.random.set_seed(random.seed(f'GapAttack tensorflow seed for seed {seed}'))

        # initial learning state
        init_state = self._init_learner_state(seed)

        # prepare callback function, will be called after each generation
        def callback_hook(_cma: CMA, logger: logging.Logger) -> None:
            if _cma.generation % (self.abort_iter // 10) == 0:
                logger.info(f'generation {_cma.generation:n}/{self.abort_iter:n} '
                            f'({_cma.generation/self.abort_iter:.1%})')
            self.sigma_history.append(_cma.σ.numpy())
            self.fitness_history.append(_cma.best_fitness())
            early_stopping = self.early_stop(_cma)
            if early_stopping:
                raise EarlyStop(early_stopping)
            if callback:
                callback(_cma, _cma.m[:-1], _cma.m[-1])

        # initialize learner
        cma = CMA(
            initial_solution=init_state,
            initial_step_size=self.sigma,
            fitness_function=objective,
            termination_no_effect=self.abort_delta,
            population_size=self.pop_size,
            callback_function=callback_hook,
            **(cma_kwargs or {}),
        )

        # with tf.device('/GPU:0'):
        # learn
        try:
            w, fitness = cma.search(max_generations=self.abort_iter)
            early_stop_reasons = []
        except EarlyStop as e:
            w, fitness = cma.best_solution(), cma.best_fitness()
            early_stop_reasons = e.reasons

        # extract results
        weights, eps = w[:-1], w[-1]
        termination_reasons = [key for key, val in cma.should_terminate(True)[1].items() if val]

        return weights, eps, fitness, termination_reasons, early_stop_reasons, cma.generation

    def discard_chain(self, weights: np.ndarray, fitness: float, pool: List[np.ndarray]) -> List[str]:
        discard_reasons = []

        # normalize weights to have positive first weight
        if weights[0] < 0:
            weights *= -1

        # check if fitness is sufficient (below)
        if fitness >= self.fitness_threshold or np.isnan(fitness) or np.isinf(fitness):
            discard_reasons.append('fitness too high')

        # Check if learned model (w) is a 'new' chain (not correlated to other chains)
        for i, v in enumerate(pool):
            corr = tf.abs(pearsonr(weights, v)[0])
            if corr > self.pool_threshold:
                discard_reasons.append(f'too similar, to learned chain {i} (correlation {corr.numpy():.2f})')

        return discard_reasons
